"""Background Removal agent — rembg-based transparent PNG output.

Bypasses Prompt Engineering (no generation prompt needed — this agent
transforms client-supplied reference images, not generates new ones). The
graph routes background_removal directly from brief_clarification here.

Reads: state.reference_image_urls (uploaded by the client at order time).
Writes: transparent PNG deliverables, one per input image.

rembg downloads the ONNX model on first run (~150 MB). Subsequent calls use
the cached model. In production this means the first order is slow (~30s);
subsequent ones are fast (~2-5s per image).
"""

from __future__ import annotations

import asyncio

import httpx

from agency.agents.base import Agent
from agency.config import Settings
from agency.db import Database
from agency.lifecycle import AgentRun
from agency.state import WorkflowState
from agency.storage.supabase_storage import SupabaseStorage


class BackgroundRemovalAgent(Agent):
    """Remove backgrounds from client-supplied reference images using rembg."""

    agent_key = "background_removal"

    def __init__(
        self,
        db: Database,
        storage: SupabaseStorage,
        settings: Settings,
    ) -> None:
        super().__init__(db)
        self._storage = storage
        self._bucket = settings.storage_bucket_deliverables
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    async def execute(self, state: WorkflowState, run: AgentRun) -> WorkflowState:
        urls = state.get("reference_image_urls") or []
        if not urls:
            raise RuntimeError(
                "BackgroundRemovalAgent requires reference images — "
                "client must upload at least one image with the order."
            )

        order_id = state["order_id"]
        run.set_input(num_inputs=len(urls))

        deliverable_ids: list[str] = []
        for idx, url in enumerate(urls):
            # Download the client's image
            resp = await self._http.get(url)
            resp.raise_for_status()
            input_blob = resp.content

            # Remove background (CPU-bound — run in thread)
            output_blob = await asyncio.to_thread(_remove_background, input_blob)

            # Upload transparent PNG
            path = f"{order_id}/nobg-{idx + 1}.png"
            stored = await self._storage.upload(
                bucket=self._bucket,
                path=path,
                data=output_blob,
                content_type="image/png",
                upsert=True,
            )
            d_id = await self.db.create_deliverable(
                order_id=order_id,
                produced_by_agent_id=run.agent_id,
                produced_by_run_id=run.run_id,
                file_url=stored.signed_url,
                file_name=f"nobg-{idx + 1}.png",
                file_type="image/png",
                file_size_bytes=len(output_blob),
                dimensions=None,   # rembg preserves source dimensions
                variant_index=idx,
                metadata={"storage_path": stored.path, "source_url": url},
            )
            deliverable_ids.append(str(d_id))

        run.set_output(deliverable_ids=deliverable_ids, count=len(deliverable_ids))
        run.log(f"Removed background from {len(deliverable_ids)} image(s)")
        return {"deliverable_ids": deliverable_ids}  # type: ignore[typeddict-item]

    async def aclose(self) -> None:
        await self._http.aclose()


def _remove_background(blob: bytes) -> bytes:
    """rembg inference. Blocking — caller must use asyncio.to_thread."""
    from rembg import remove  # lazy import — heavy onnxruntime dep
    return remove(blob)
