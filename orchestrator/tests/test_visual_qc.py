"""Unit tests for the Visual QC agent.

Mocks AnthropicClient.complete_json_with_image and SupabaseStorage.download.
Verifies:
  * Skips deliverables that failed Technical QC (no wasted vision tokens)
  * Per-variant scoring writes to deliverables.quality_score
  * Threshold gating: zero passing variants raises
  * Score clamping to [0.0, 1.0]
  * Cost accumulation across variants
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from agency.agents.visual_qc import VisualQC
from agency.storage.supabase_storage import SupabaseStorage
from tests.conftest import make_state

ORDER_ID = UUID("00000000-0000-0000-0000-0000000000b0")
D1, D2 = UUID("00000000-0000-0000-0000-00000000a001"), UUID("00000000-0000-0000-0000-00000000a002")


def _deliv(id_: UUID, *, qc: bool, path: str | None = "p.jpg") -> dict:
    return {
        "id": str(id_),
        "technical_qc_passed": qc,
        "metadata": {"storage_path": path} if path else {},
        "variant_index": 0,
    }


@pytest.fixture
def mock_storage() -> MagicMock:
    s = MagicMock(spec=SupabaseStorage)
    s.download = AsyncMock(return_value=b"\xff\xd8\xff\xe0fakejpeg")
    return s


@pytest.fixture
def visual_qc(mock_db, mock_anthropic, mock_storage, settings) -> VisualQC:
    mock_db.get_agent_by_key = AsyncMock(
        return_value={"id": "00000000-0000-0000-0000-0000000000aa", "agent_key": "visual_qc", "layer": "quality"}
    )
    mock_db.update_deliverable_qc = AsyncMock()
    # Replace mock_anthropic.complete_json with the vision variant
    mock_anthropic.complete_json_with_image = AsyncMock()
    return VisualQC(db=mock_db, anthropic=mock_anthropic, storage=mock_storage, settings=settings)


async def test_skips_technically_failed_deliverables(
    visual_qc, mock_db, mock_anthropic, mock_storage, completion_result_factory
):
    mock_db.list_deliverables = AsyncMock(
        return_value=[_deliv(D1, qc=True), _deliv(D2, qc=False)]
    )
    mock_anthropic.complete_json_with_image.return_value = (
        {"quality_score": 0.85, "issues": [], "publishable": True, "rationale": "fine"},
        completion_result_factory(cost_usd=0.004),
    )

    await visual_qc(make_state(order_id=ORDER_ID, service_type="thumbnail"))

    # Only the technically-passing variant was scored
    assert mock_anthropic.complete_json_with_image.await_count == 1
    assert mock_storage.download.await_count == 1


async def test_writes_quality_score_per_variant(
    visual_qc, mock_db, mock_anthropic, completion_result_factory
):
    mock_db.list_deliverables = AsyncMock(
        return_value=[_deliv(D1, qc=True), _deliv(D2, qc=True)]
    )
    mock_anthropic.complete_json_with_image.side_effect = [
        ({"quality_score": 0.92, "publishable": True, "issues": [], "rationale": "x"}, completion_result_factory(cost_usd=0.004)),
        ({"quality_score": 0.55, "publishable": False, "issues": ["distorted hand"], "rationale": "y"}, completion_result_factory(cost_usd=0.004)),
    ]

    await visual_qc(make_state(order_id=ORDER_ID, service_type="thumbnail"))

    # Both got scored
    assert mock_db.update_deliverable_qc.await_count == 2
    scores = {
        c.args[0]: c.kwargs["quality_score"]
        for c in mock_db.update_deliverable_qc.await_args_list
    }
    assert scores[D1] == pytest.approx(0.92)
    assert scores[D2] == pytest.approx(0.55)


async def test_raises_when_no_variant_meets_threshold(
    visual_qc, mock_db, mock_anthropic, completion_result_factory
):
    mock_db.list_deliverables = AsyncMock(return_value=[_deliv(D1, qc=True)])
    mock_anthropic.complete_json_with_image.return_value = (
        {"quality_score": 0.40, "publishable": False, "issues": ["broken face"], "rationale": "bad"},
        completion_result_factory(cost_usd=0.004),
    )

    with pytest.raises(RuntimeError, match="no variants reached threshold"):
        await visual_qc(make_state(order_id=ORDER_ID, service_type="thumbnail"))


async def test_raises_when_no_technically_passing_candidates(
    visual_qc, mock_db, mock_anthropic
):
    mock_db.list_deliverables = AsyncMock(return_value=[_deliv(D1, qc=False)])

    with pytest.raises(RuntimeError, match="no technically-passing"):
        await visual_qc(make_state(order_id=ORDER_ID, service_type="thumbnail"))

    mock_anthropic.complete_json_with_image.assert_not_called()


async def test_clamps_out_of_range_scores(
    visual_qc, mock_db, mock_anthropic, completion_result_factory
):
    """Claude can return 1.2 or -0.1; the agent must clamp to [0, 1]."""
    mock_db.list_deliverables = AsyncMock(
        return_value=[_deliv(D1, qc=True), _deliv(D2, qc=True)]
    )
    mock_anthropic.complete_json_with_image.side_effect = [
        ({"quality_score": 1.5, "publishable": True, "issues": [], "rationale": "x"}, completion_result_factory()),
        ({"quality_score": -0.2, "publishable": False, "issues": [], "rationale": "y"}, completion_result_factory()),
    ]

    await visual_qc(make_state(order_id=ORDER_ID, service_type="thumbnail"))

    scores = {
        c.args[0]: c.kwargs["quality_score"]
        for c in mock_db.update_deliverable_qc.await_args_list
    }
    assert scores[D1] == 1.0  # clamped from 1.5
    assert scores[D2] == 0.0  # clamped from -0.2


async def test_cost_accumulates_across_variants(
    visual_qc, mock_db, mock_anthropic, completion_result_factory
):
    mock_db.list_deliverables = AsyncMock(
        return_value=[_deliv(D1, qc=True), _deliv(D2, qc=True)]
    )
    mock_anthropic.complete_json_with_image.side_effect = [
        ({"quality_score": 0.9, "publishable": True, "issues": [], "rationale": "x"}, completion_result_factory(cost_usd=0.004)),
        ({"quality_score": 0.8, "publishable": True, "issues": [], "rationale": "y"}, completion_result_factory(cost_usd=0.005)),
    ]

    await visual_qc(make_state(order_id=ORDER_ID, service_type="thumbnail"))

    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    assert finish_kwargs["cost_usd"] == pytest.approx(0.009)


async def test_delivery_packager_respects_quality_score(settings, mock_db):
    """Packager's filter rejects low-quality variants even if technically passing."""
    from agency.agents.delivery_packager import DeliveryPackager
    from agency.storage.supabase_storage import SupabaseStorage

    mock_db.get_agent_by_key = AsyncMock(
        return_value={"id": "00000000-0000-0000-0000-0000000000aa", "agent_key": "delivery_packager", "layer": "delivery"}
    )
    mock_db.list_deliverables = AsyncMock(
        return_value=[
            {
                "id": str(D1),
                "technical_qc_passed": True,
                "quality_score": 0.4,  # below 0.70 threshold
                "metadata": {"storage_path": "p.jpg"},
                "variant_index": 0,
                "parent_deliverable_id": None,
                "file_name": "x.jpg",
            },
        ]
    )
    mock_db.update_deliverable_qc = AsyncMock()
    mock_db.update_order_status = AsyncMock()
    mock_db.create_delivery_package = AsyncMock()

    mock_storage = MagicMock(spec=SupabaseStorage)
    packager = DeliveryPackager(db=mock_db, storage=mock_storage, settings=settings)

    with pytest.raises(RuntimeError, match="no QC-passed"):
        await packager(make_state(order_id=ORDER_ID, service_type="thumbnail"))


async def test_delivery_packager_passes_high_quality(settings, mock_db):
    """Variants with quality_score >= threshold are packaged normally."""
    from agency.agents.delivery_packager import DeliveryPackager
    from agency.storage.supabase_storage import StoredFile, SupabaseStorage

    mock_db.get_agent_by_key = AsyncMock(
        return_value={"id": "00000000-0000-0000-0000-0000000000aa", "agent_key": "delivery_packager", "layer": "delivery"}
    )
    mock_db.list_deliverables = AsyncMock(
        return_value=[
            {
                "id": str(D1),
                "technical_qc_passed": True,
                "quality_score": 0.85,  # well above 0.70
                "metadata": {"storage_path": "p.jpg"},
                "variant_index": 0,
                "parent_deliverable_id": None,
                "file_name": "x.jpg",
            },
        ]
    )
    mock_db.update_deliverable_qc = AsyncMock()
    mock_db.update_order_status = AsyncMock()
    mock_db.create_delivery_package = AsyncMock(return_value=UUID("00000000-0000-0000-0000-00000000c0c0"))

    mock_storage = MagicMock(spec=SupabaseStorage)
    mock_storage.download = AsyncMock(return_value=b"jpegbytes")

    async def _upload(*, bucket, path, data, content_type, upsert=False, ttl_seconds=None):
        return StoredFile(bucket=bucket, path=path, signed_url="https://x", expires_in_seconds=ttl_seconds or 604_800)

    mock_storage.upload = AsyncMock(side_effect=_upload)

    packager = DeliveryPackager(db=mock_db, storage=mock_storage, settings=settings)
    result = await packager(make_state(order_id=ORDER_ID, service_type="thumbnail"))

    assert result["package_id"] is not None
    mock_db.create_delivery_package.assert_awaited_once()
