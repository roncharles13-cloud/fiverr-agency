"""Logo Generator — raster concept generation + SVG vectorization.

Two-stage per variant:
  1. Generate with Flux (square_hd / 1024x1024, solid background, flat shapes).
  2. Vectorize the raster to SVG via vtracer.
  3. Upload both files; create a deliverable row for each.

Clients receive the raster (for preview / social use) and the SVG (for printing,
embroidery, scaling to any size). Both land in the ZIP via Delivery Packager.

Design notes:
  * 2 variants, not 3 — logo design is quality-over-quantity.
  * No parent_deliverable_id on either file so Delivery Packager includes both.
  * Raster → Technical QC → Visual QC (normal path).
  * SVG → Technical QC sets technical_qc_passed=True by MIME type (Pillow
    cannot open SVG; see technical_qc.py).
  * Vectorization is CPU-bound; run in asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import os
import tempfile

from agency.agents.base import Agent
from agency.clients.fal_client import FalClient
from agency.config import Settings
from agency.db import Database
from agency.lifecycle import AgentRun
from agency.state import WorkflowState
from agency.storage.supabase_storage import SupabaseStorage

_IMAGE_SIZE = "square_hd"   # 1024x1024 -- best for flat logo shapes
_NUM_VARIANTS = 2
_VTRACER_PARAMS = {
    "colormode": "color",
    "hierarchical": "stacked",
    "mode": "spline",
    "filter_speckle": 4,
    "color_precision": 6,
    "layer_difference": 16,
    "corner_threshold": 60,
    "length_threshold": 4.0,
    "max_iterations": 10,
    "splice_threshold": 45,
    "path_precision": 3,
}


class LogoGenerator(Agent):
    """Generate raster logo concepts and vectorize each to SVG."""

    agent_key = "logo_gen"

    def __init__(
        self,
        db: Database,
        fal: FalClient,
        storage: SupabaseStorage,
        settings: Settings,
    ) -> None:
        super().__init__(db)
        self._fal = fal
        self._storage = storage
        self._bucket = settings.storage_bucket_deliverables

    async def execute(self, state: WorkflowState, run: AgentRun) -> WorkflowState:
        prompt = state.get("refined_prompt")
        if not prompt:
            raise ValueError("LogoGenerator requires state['refined_prompt']")

        order_id = state["order_id"]
        run.set_input(prompt=prompt, num_variants=_NUM_VARIANTS, image_size=_IMAGE_SIZE)

        generation = await self._fal.generate(
            prompt=prompt,
            negative_prompt=state.get("negative_prompt"),
            num_images=_NUM_VARIANTS,
            image_size=_IMAGE_SIZE,
        )
        run.add_cost(generation.cost_usd)

        if not generation.images:
            raise RuntimeError("fal.ai returned zero images — safety filter triggered")

        deliverable_ids: list[str] = []
        for idx, image in enumerate(generation.images):
            raster_blob = await self._fal.download(image.url)

            # Upload raster (JPEG) for QC + preview
            raster_path = f"{order_id}/logo-{idx + 1}.jpg"
            raster_stored = await self._storage.upload(
                bucket=self._bucket,
                path=raster_path,
                data=raster_blob,
                content_type="image/jpeg",
                upsert=True,
            )
            raster_id = await self.db.create_deliverable(
                order_id=order_id,
                produced_by_agent_id=run.agent_id,
                produced_by_run_id=run.run_id,
                file_url=raster_stored.signed_url,
                file_name=f"logo-{idx + 1}.jpg",
                file_type="image/jpeg",
                file_size_bytes=len(raster_blob),
                dimensions={"width": image.width, "height": image.height, "dpi": 72},
                variant_index=idx,
                metadata={"storage_path": raster_stored.path, "fal_seed": generation.seed},
            )
            deliverable_ids.append(str(raster_id))

            # Vectorize → SVG
            svg_bytes = await asyncio.to_thread(_vectorize, raster_blob)
            svg_path = f"{order_id}/logo-{idx + 1}.svg"
            svg_stored = await self._storage.upload(
                bucket=self._bucket,
                path=svg_path,
                data=svg_bytes,
                content_type="image/svg+xml",
                upsert=True,
            )
            svg_id = await self.db.create_deliverable(
                order_id=order_id,
                produced_by_agent_id=run.agent_id,
                produced_by_run_id=run.run_id,
                file_url=svg_stored.signed_url,
                file_name=f"logo-{idx + 1}.svg",
                file_type="image/svg+xml",
                file_size_bytes=len(svg_bytes),
                dimensions={"width": image.width, "height": image.height, "dpi": 72},
                variant_index=idx,
                metadata={"storage_path": svg_stored.path, "vectorized_from": str(raster_id)},
            )
            deliverable_ids.append(str(svg_id))

        run.set_output(deliverable_ids=deliverable_ids, cost_usd=generation.cost_usd)
        run.log(f"Generated {_NUM_VARIANTS} logo variants + SVGs (${generation.cost_usd:.2f})")
        return {"deliverable_ids": deliverable_ids}  # type: ignore[typeddict-item]


def _vectorize(raster_blob: bytes) -> bytes:
    """Convert raster PNG/JPEG bytes → SVG bytes via vtracer. Blocking — use to_thread."""
    import vtracer  # lazy import — not installed in test environments

    with (
        tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fin,
        tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as fout,
    ):
        fin_path, fout_path = fin.name, fout.name
        fin.write(raster_blob)

    try:
        vtracer.convert_raw_image_to_svg(fin_path, fout_path, **_VTRACER_PARAMS)
        with open(fout_path, "rb") as f:
            return f.read()
    finally:
        os.unlink(fin_path)
        os.unlink(fout_path)
