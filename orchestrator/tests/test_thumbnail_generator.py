"""Unit tests for the Thumbnail Generator agent.

Mocks `FalClient` and `SupabaseStorage`. Verifies:
  * Generates the requested number of variations
  * Downloads each, uploads each, creates one deliverable row per variant
  * Records cost on the run
  * Returns deliverable_ids in state
  * Fails gracefully when prerequisites are missing or fal returns 0 images
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from agency.agents.thumbnail_generator import ThumbnailGenerator
from agency.clients.fal_client import FalClient, GeneratedImage, GenerationResult
from agency.storage.supabase_storage import StoredFile, SupabaseStorage
from tests.conftest import make_state

DELIVERABLE_IDS = [
    UUID("00000000-0000-0000-0000-00000000d001"),
    UUID("00000000-0000-0000-0000-00000000d002"),
    UUID("00000000-0000-0000-0000-00000000d003"),
]


@pytest.fixture
def mock_fal() -> MagicMock:
    fal = MagicMock(spec=FalClient)
    fal.endpoint = "fal-ai/flux-pro/v1.1"
    fal.generate = AsyncMock()
    fal.download = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    return fal


@pytest.fixture
def mock_storage() -> MagicMock:
    storage = MagicMock(spec=SupabaseStorage)

    async def _upload(*, bucket, path, data, content_type, upsert=False, ttl_seconds=None):
        return StoredFile(
            bucket=bucket,
            path=path,
            signed_url=f"https://supabase.example/{bucket}/{path}?signed",
            expires_in_seconds=ttl_seconds or 604_800,
        )

    storage.upload = AsyncMock(side_effect=_upload)
    return storage


@pytest.fixture
def thumbnail_gen(mock_db, mock_fal, mock_storage, settings) -> ThumbnailGenerator:
    mock_db.get_agent_by_key = AsyncMock(
        return_value={
            "id": "00000000-0000-0000-0000-0000000000aa",
            "agent_key": "thumbnail_gen",
            "display_name": "Thumbnail Generator",
            "layer": "generation",
        }
    )
    mock_db.create_deliverable = AsyncMock(side_effect=DELIVERABLE_IDS)
    return ThumbnailGenerator(db=mock_db, fal=mock_fal, storage=mock_storage, settings=settings)


def _gen_result(n: int = 3) -> GenerationResult:
    return GenerationResult(
        images=tuple(
            GeneratedImage(
                url=f"https://fal.media/img-{i}.jpg",
                width=1280,
                height=720,
                content_type="image/jpeg",
            )
            for i in range(n)
        ),
        seed=42,
        cost_usd=0.12,
        prompt="cinematic thumbnail",
    )


# ── Happy path ──────────────────────────────────────────────────────────────


async def test_generates_three_variations_and_records_each(
    thumbnail_gen, mock_db, mock_fal, mock_storage
):
    mock_fal.generate.return_value = _gen_result(n=3)

    state = make_state(
        service_type="thumbnail",
        refined_prompt="cinematic thumbnail, neon, shocked face",
        negative_prompt="blurry, watermark",
    )
    result = await thumbnail_gen(state)

    # fal.generate called with the right shape
    gen_kwargs = mock_fal.generate.await_args.kwargs
    assert gen_kwargs["prompt"] == "cinematic thumbnail, neon, shocked face"
    assert gen_kwargs["negative_prompt"] == "blurry, watermark"
    assert gen_kwargs["num_images"] == 3
    assert gen_kwargs["image_size"] == "landscape_16_9"

    # Three downloads + uploads + deliverables
    assert mock_fal.download.await_count == 3
    assert mock_storage.upload.await_count == 3
    assert mock_db.create_deliverable.await_count == 3

    # variant_index increments 0,1,2
    variant_indices = [
        call.kwargs["variant_index"] for call in mock_db.create_deliverable.await_args_list
    ]
    assert variant_indices == [0, 1, 2]

    # Dimensions stored
    first = mock_db.create_deliverable.await_args_list[0].kwargs
    assert first["dimensions"] == {"width": 1280, "height": 720, "dpi": 72}
    assert first["file_type"] == "image/jpeg"

    # State update carries deliverable_ids as strings
    assert len(result["deliverable_ids"]) == 3
    assert all(isinstance(d, str) for d in result["deliverable_ids"])

    # Cost was recorded
    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    assert finish_kwargs["cost_usd"] == pytest.approx(0.12)


# ── Failure modes ──────────────────────────────────────────────────────────


async def test_missing_refined_prompt_raises(thumbnail_gen, mock_db, mock_fal):
    """The agent must not call fal.ai if state.refined_prompt is missing."""
    state = make_state(service_type="thumbnail")  # no refined_prompt

    with pytest.raises(ValueError, match="refined_prompt"):
        await thumbnail_gen(state)

    mock_fal.generate.assert_not_called()

    # Lifecycle recorded the error
    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    assert finish_kwargs["status"] == "error"


async def test_zero_images_returned_raises(thumbnail_gen, mock_db, mock_fal, mock_storage):
    """Safety-filter rejection or fal anomaly — must surface, not silently succeed."""
    mock_fal.generate.return_value = _gen_result(n=0)

    state = make_state(refined_prompt="some prompt")
    with pytest.raises(RuntimeError, match="zero images"):
        await thumbnail_gen(state)

    mock_storage.upload.assert_not_called()
    mock_db.create_deliverable.assert_not_called()

    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    assert finish_kwargs["status"] == "error"


async def test_storage_upload_uses_order_scoped_path(
    thumbnail_gen, mock_db, mock_fal, mock_storage
):
    """Deliverables must be namespaced by order_id so a buyer can't see another's files."""
    mock_fal.generate.return_value = _gen_result(n=1)
    order_id = UUID("00000000-0000-0000-0000-0000000000b0")

    state = make_state(refined_prompt="x", order_id=order_id)
    await thumbnail_gen(state)

    upload_kwargs = mock_storage.upload.await_args.kwargs
    assert upload_kwargs["path"].startswith(f"{order_id}/")
    assert upload_kwargs["bucket"] == "deliverables"
    assert upload_kwargs["upsert"] is True  # retries can safely rewrite the same path
