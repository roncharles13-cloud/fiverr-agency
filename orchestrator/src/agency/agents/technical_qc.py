"""Technical QC agent.

Verifies each deliverable meets format/size requirements before it can be
packaged. Pure technical checks — no LLM, no vision model.

Checks:
  * Image format is JPEG or PNG (not WEBP — Fiverr clients may not have viewers)
  * Dimensions within tolerance of expected size for the service_type
  * File size below 10 MB
  * No truncated / corrupted image data

Side effects: updates `deliverables.technical_qc_passed` and writes per-failure
reasons to `agent_runs.output_data` for operator visibility.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from agency.agents.base import Agent
from agency.db import Database
from agency.editing import inspect_image
from agency.lifecycle import AgentRun
from agency.state import ServiceType, WorkflowState
from agency.storage.supabase_storage import SupabaseStorage

# Expected (width, height) per service. ±5% tolerance.
# (0, 0) skips dimension checking — used for services with variable aspect
# ratios where the generator's own preset is the source of truth.
_EXPECTED_DIMENSIONS: dict[ServiceType, tuple[int, int]] = {
    "thumbnail": (1024, 576),     # fal Flux Pro 1.1 landscape_16_9 actual output
    "social_graphic": (0, 0),    # multi-platform: square / portrait / landscape
    "headshot": (768, 1024),      # portrait_4_3 preset
    "logo": (1024, 1024),
    "business_design": (1024, 576),
    "background_removal": (0, 0), # variable input → variable output
}

_DIMENSION_TOLERANCE = 0.05
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
_ALLOWED_FORMATS = {"JPEG", "PNG"}


class TechnicalQC(Agent):
    """Inspect each deliverable; set technical_qc_passed."""

    agent_key = "technical_qc"

    def __init__(
        self,
        db: Database,
        storage: SupabaseStorage,
        settings: Any,  # Settings — kept loose to avoid circular import gymnastics
    ) -> None:
        super().__init__(db)
        self._storage = storage
        self._bucket = settings.storage_bucket_deliverables

    async def execute(self, state: WorkflowState, run: AgentRun) -> WorkflowState:
        order_id = state["order_id"]
        deliverables = await self.db.list_deliverables(order_id)
        if not deliverables:
            raise RuntimeError("technical_qc found no deliverables")

        expected = _EXPECTED_DIMENSIONS.get(state["service_type"])
        run.set_input(num_deliverables=len(deliverables), expected_dimensions=expected)

        per_deliverable: list[dict[str, Any]] = []
        pass_count = 0
        for d in deliverables:
            # SVG files cannot be opened by Pillow — mark passed by MIME type.
            if d.get("file_type") == "image/svg+xml":
                await self.db.update_deliverable_qc(UUID(d["id"]), technical_qc_passed=True)
                per_deliverable.append({"deliverable_id": d["id"], "passed": True, "svg_auto_pass": True})
                pass_count += 1
                continue

            path = (d.get("metadata") or {}).get("storage_path")
            if not path:
                per_deliverable.append(
                    {"deliverable_id": d["id"], "passed": False, "reason": "no storage_path"}
                )
                await self.db.update_deliverable_qc(UUID(d["id"]), technical_qc_passed=False)
                continue

            blob = await self._storage.download(bucket=self._bucket, path=path)
            failures = _check(blob, expected=expected)
            passed = not failures
            if passed:
                pass_count += 1
            per_deliverable.append(
                {"deliverable_id": d["id"], "passed": passed, "failures": failures}
            )
            await self.db.update_deliverable_qc(UUID(d["id"]), technical_qc_passed=passed)

        run.set_output(checked=len(deliverables), passed=pass_count, results=per_deliverable)
        run.log(f"{pass_count}/{len(deliverables)} deliverables passed technical QC")

        if pass_count == 0:
            raise RuntimeError("technical_qc: every deliverable failed — manual review required")

        return {"qc_passed": True}  # type: ignore[typeddict-item]


# ── Pure helpers ────────────────────────────────────────────────────────────


def _check(blob: bytes, *, expected: tuple[int, int] | None) -> list[str]:
    """Run all technical checks. Returns a list of failure reasons (empty = pass)."""
    failures: list[str] = []

    if len(blob) > _MAX_FILE_SIZE_BYTES:
        failures.append(f"file too large: {len(blob)} bytes > {_MAX_FILE_SIZE_BYTES}")

    try:
        info = inspect_image(blob)
    except Exception as exc:
        failures.append(f"image decode failed: {type(exc).__name__}: {exc}")
        return failures

    if info.format not in _ALLOWED_FORMATS:
        failures.append(f"format {info.format} not in {sorted(_ALLOWED_FORMATS)}")

    if expected and expected != (0, 0):
        ew, eh = expected
        w_tol = ew * _DIMENSION_TOLERANCE
        h_tol = eh * _DIMENSION_TOLERANCE
        if abs(info.width - ew) > w_tol or abs(info.height - eh) > h_tol:
            failures.append(
                f"dimensions {info.width}x{info.height} outside ±5% of {ew}x{eh}"
            )

    return failures
