"""Unit tests for Text Renderer, Technical QC, and Delivery Packager.

All three exercise the storage + DB layers — both are mocked. The pure
editing helpers are covered separately in `test_editing.py`.
"""

from __future__ import annotations

import io
import zipfile
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from PIL import Image

from agency.agents.delivery_packager import DeliveryPackager, _draft_message
from agency.agents.technical_qc import TechnicalQC
from agency.agents.text_renderer import TextRenderer
from agency.storage.supabase_storage import StoredFile, SupabaseStorage
from tests.conftest import make_state

ORDER_ID = UUID("00000000-0000-0000-0000-0000000000b0")
PARENT_IDS = [
    UUID("00000000-0000-0000-0000-00000000a001"),
    UUID("00000000-0000-0000-0000-00000000a002"),
]
PACKAGE_ID = UUID("00000000-0000-0000-0000-00000000c0c0")


def _jpeg(w: int, h: int) -> bytes:
    img = Image.new("RGB", (w, h), (50, 100, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _deliverable(
    *,
    id_: UUID,
    variant: int,
    parent: UUID | None = None,
    storage_path: str | None = None,
    qc_passed: bool | None = None,
) -> dict:
    return {
        "id": str(id_),
        "order_id": str(ORDER_ID),
        "variant_index": variant,
        "parent_deliverable_id": str(parent) if parent else None,
        "metadata": {"storage_path": storage_path or f"{ORDER_ID}/thumbnail-{variant + 1}.jpg"},
        "file_name": f"thumbnail-{variant + 1}.jpg",
        "dimensions": {"width": 1280, "height": 720, "dpi": 72},
        "technical_qc_passed": qc_passed,
    }


@pytest.fixture
def mock_storage() -> MagicMock:
    storage = MagicMock(spec=SupabaseStorage)
    storage.download = AsyncMock(return_value=_jpeg(1280, 720))

    async def _upload(*, bucket, path, data, content_type, upsert=False, ttl_seconds=None):
        return StoredFile(
            bucket=bucket,
            path=path,
            signed_url=f"https://supabase.example/{bucket}/{path}?signed",
            expires_in_seconds=ttl_seconds or 604_800,
        )

    storage.upload = AsyncMock(side_effect=_upload)
    return storage


# ============================================================================
# Text Renderer
# ============================================================================


@pytest.fixture
def text_renderer(mock_db, mock_storage, settings) -> TextRenderer:
    mock_db.get_agent_by_key = AsyncMock(
        return_value={"id": "00000000-0000-0000-0000-0000000000aa", "agent_key": "text_renderer", "layer": "editing"}
    )
    mock_db.list_deliverables = AsyncMock(
        return_value=[
            _deliverable(id_=PARENT_IDS[0], variant=0),
            _deliverable(id_=PARENT_IDS[1], variant=1),
        ]
    )
    mock_db.create_deliverable = AsyncMock(
        side_effect=[
            UUID("00000000-0000-0000-0000-00000000a101"),
            UUID("00000000-0000-0000-0000-00000000a102"),
        ]
    )
    return TextRenderer(db=mock_db, storage=mock_storage, settings=settings)


async def test_text_renderer_skips_when_no_overlay(text_renderer, mock_db, mock_storage):
    state = make_state(order_id=ORDER_ID)  # no text_overlay
    await text_renderer(state)

    mock_storage.download.assert_not_called()
    mock_db.create_deliverable.assert_not_called()
    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    assert finish_kwargs["log_summary"] == "no overlay requested"


async def test_text_renderer_composites_each_original(text_renderer, mock_db, mock_storage):
    state = make_state(order_id=ORDER_ID, text_overlay="INSANE WIN")
    result = await text_renderer(state)

    # Both originals downloaded + composited + uploaded + recorded
    assert mock_storage.download.await_count == 2
    assert mock_storage.upload.await_count == 2
    assert mock_db.create_deliverable.await_count == 2

    # New deliverables link back to parents
    create_calls = mock_db.create_deliverable.await_args_list
    parents = {c.kwargs["parent_deliverable_id"] for c in create_calls}
    assert parents == set(PARENT_IDS)

    # State carries the new ids
    assert len(result["deliverable_ids"]) == 2


async def test_text_renderer_skips_existing_overlay_deliverables(
    text_renderer, mock_db, mock_storage
):
    """Rows that already have parent_deliverable_id are themselves overlays — skip."""
    mock_db.list_deliverables.return_value = [
        _deliverable(id_=PARENT_IDS[0], variant=0),
        _deliverable(id_=PARENT_IDS[1], variant=0, parent=PARENT_IDS[0]),  # overlay of #0
    ]
    mock_db.create_deliverable = AsyncMock(return_value=UUID("00000000-0000-0000-0000-00000000a999"))

    state = make_state(order_id=ORDER_ID, text_overlay="HI")
    await text_renderer(state)

    # Only the original was processed
    assert mock_storage.download.await_count == 1
    assert mock_db.create_deliverable.await_count == 1


# ============================================================================
# Technical QC
# ============================================================================


@pytest.fixture
def technical_qc(mock_db, mock_storage, settings) -> TechnicalQC:
    mock_db.get_agent_by_key = AsyncMock(
        return_value={"id": "00000000-0000-0000-0000-0000000000aa", "agent_key": "technical_qc", "layer": "quality"}
    )
    mock_db.update_deliverable_qc = AsyncMock()
    return TechnicalQC(db=mock_db, storage=mock_storage, settings=settings)


async def test_technical_qc_marks_passing_thumbnails(technical_qc, mock_db, mock_storage):
    mock_db.list_deliverables = AsyncMock(
        return_value=[
            _deliverable(id_=PARENT_IDS[0], variant=0),
            _deliverable(id_=PARENT_IDS[1], variant=1),
        ]
    )
    mock_storage.download.return_value = _jpeg(1280, 720)

    state = make_state(order_id=ORDER_ID, service_type="thumbnail")
    result = await technical_qc(state)

    assert result["qc_passed"] is True
    # Both deliverables got marked passing
    assert mock_db.update_deliverable_qc.await_count == 2
    for call in mock_db.update_deliverable_qc.await_args_list:
        assert call.kwargs["technical_qc_passed"] is True


async def test_technical_qc_marks_wrong_size_failing(technical_qc, mock_db, mock_storage):
    mock_db.list_deliverables = AsyncMock(
        return_value=[_deliverable(id_=PARENT_IDS[0], variant=0)]
    )
    # Way off from 1280x720
    mock_storage.download.return_value = _jpeg(640, 360)

    state = make_state(order_id=ORDER_ID, service_type="thumbnail")
    with pytest.raises(RuntimeError, match="every deliverable failed"):
        await technical_qc(state)

    call = mock_db.update_deliverable_qc.await_args_list[0]
    assert call.kwargs["technical_qc_passed"] is False


async def test_technical_qc_records_per_deliverable_results(technical_qc, mock_db, mock_storage):
    """One passing + one failing — partial pass advances the pipeline."""
    mock_db.list_deliverables = AsyncMock(
        return_value=[
            _deliverable(id_=PARENT_IDS[0], variant=0),
            _deliverable(id_=PARENT_IDS[1], variant=1, storage_path="bad.jpg"),
        ]
    )
    # First call passes, second fails (wrong dimensions)
    mock_storage.download = AsyncMock(side_effect=[_jpeg(1280, 720), _jpeg(640, 360)])

    state = make_state(order_id=ORDER_ID, service_type="thumbnail")
    await technical_qc(state)

    # Per-deliverable results captured for operator visibility
    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    out = finish_kwargs["output_data"]
    assert out["checked"] == 2
    assert out["passed"] == 1


# ============================================================================
# Delivery Packager
# ============================================================================


@pytest.fixture
def delivery_packager(mock_db, mock_storage, settings) -> DeliveryPackager:
    mock_db.get_agent_by_key = AsyncMock(
        return_value={"id": "00000000-0000-0000-0000-0000000000aa", "agent_key": "delivery_packager", "layer": "delivery"}
    )
    mock_db.update_deliverable_qc = AsyncMock()
    mock_db.update_order_status = AsyncMock()
    mock_db.create_delivery_package = AsyncMock(return_value=PACKAGE_ID)
    return DeliveryPackager(db=mock_db, storage=mock_storage, settings=settings)


async def test_packager_prefers_overlay_variants(
    delivery_packager, mock_db, mock_storage
):
    """When overlays exist they ship, not the unbranded originals."""
    mock_db.list_deliverables = AsyncMock(
        return_value=[
            _deliverable(id_=PARENT_IDS[0], variant=0, qc_passed=True),
            _deliverable(
                id_=UUID("00000000-0000-0000-0000-00000000a101"),
                variant=0,
                parent=PARENT_IDS[0],
                storage_path=f"{ORDER_ID}/text-0.jpg",
                qc_passed=True,
            ),
        ]
    )

    state = make_state(order_id=ORDER_ID, service_type="thumbnail")
    result = await delivery_packager(state)

    # Only the overlay variant was packaged + approved
    assert mock_db.update_deliverable_qc.await_count == 1
    approved_id = mock_db.update_deliverable_qc.await_args.args[0]
    assert str(approved_id) == "00000000-0000-0000-0000-00000000a101"

    assert result["package_id"] == PACKAGE_ID


async def test_packager_falls_back_to_originals_when_no_overlays(
    delivery_packager, mock_db, mock_storage
):
    mock_db.list_deliverables = AsyncMock(
        return_value=[
            _deliverable(id_=PARENT_IDS[0], variant=0, qc_passed=True),
            _deliverable(id_=PARENT_IDS[1], variant=1, qc_passed=True),
        ]
    )

    state = make_state(order_id=ORDER_ID, service_type="thumbnail")
    await delivery_packager(state)

    # Both originals approved
    assert mock_db.update_deliverable_qc.await_count == 2


async def test_packager_raises_when_no_qc_passed(delivery_packager, mock_db):
    mock_db.list_deliverables = AsyncMock(
        return_value=[_deliverable(id_=PARENT_IDS[0], variant=0, qc_passed=False)]
    )

    state = make_state(order_id=ORDER_ID, service_type="thumbnail")
    with pytest.raises(RuntimeError, match="no QC-passed"):
        await delivery_packager(state)

    mock_db.create_delivery_package.assert_not_called()


async def test_packager_uploads_valid_zip(delivery_packager, mock_db, mock_storage):
    mock_db.list_deliverables = AsyncMock(
        return_value=[
            _deliverable(id_=PARENT_IDS[0], variant=0, qc_passed=True),
            _deliverable(id_=PARENT_IDS[1], variant=1, qc_passed=True),
        ]
    )

    state = make_state(order_id=ORDER_ID, service_type="thumbnail")
    await delivery_packager(state)

    # The bytes uploaded should be a valid ZIP containing both file names
    upload_kwargs = mock_storage.upload.await_args.kwargs
    assert upload_kwargs["bucket"] == "delivery-packages"
    assert upload_kwargs["content_type"] == "application/zip"

    with zipfile.ZipFile(io.BytesIO(upload_kwargs["data"])) as zf:
        names = set(zf.namelist())
    assert names == {"thumbnail-1.jpg", "thumbnail-2.jpg"}


async def test_packager_flips_order_status_to_ready(
    delivery_packager, mock_db
):
    mock_db.list_deliverables = AsyncMock(
        return_value=[_deliverable(id_=PARENT_IDS[0], variant=0, qc_passed=True)]
    )

    state = make_state(order_id=ORDER_ID, service_type="thumbnail")
    await delivery_packager(state)

    mock_db.update_order_status.assert_awaited_once()
    update_kwargs = mock_db.update_order_status.await_args.kwargs
    assert update_kwargs["status"] == "ready_for_delivery"


def test_draft_message_pluralizes_correctly():
    msg_single = _draft_message(service_type="thumbnail", variant_count=1)
    msg_multi = _draft_message(service_type="thumbnail", variant_count=3)
    assert "1 variation " in msg_single  # no 's'
    assert "3 variations" in msg_multi
