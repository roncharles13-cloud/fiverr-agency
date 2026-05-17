"""Unit tests for Social Graphics — aspect inference and base-class integration.

The shared generation loop is already covered by `test_thumbnail_generator.py`.
This file focuses on what's *new*: the platform-to-image_size heuristic, and a
smoke test that SocialGraphicsGenerator inherits the loop correctly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from agency.agents.social_graphics_generator import (
    SocialGraphicsGenerator,
    _infer_image_size,
)
from agency.clients.fal_client import FalClient, GeneratedImage, GenerationResult
from agency.storage.supabase_storage import StoredFile, SupabaseStorage
from tests.conftest import make_state

# ── Aspect inference ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Portrait / vertical
        ("Instagram story background", "portrait_16_9"),
        ("9:16 reel for TikTok", "portrait_16_9"),
        ("vertical poster", "portrait_16_9"),
        # Landscape
        ("Twitter header banner", "landscape_16_9"),
        ("16:9 cover photo for YouTube", "landscape_16_9"),
        ("Facebook cover photo", "landscape_16_9"),
        # Square (default)
        ("Instagram feed post", "square_hd"),
        ("just a clean square design", "square_hd"),
        ("no aspect hints at all", "square_hd"),
        ("", "square_hd"),
    ],
)
def test_infer_image_size_routes_by_keyword(text: str, expected: str):
    state = make_state(service_type="social_graphic", refined_prompt=text)
    assert _infer_image_size(state) == expected


def test_infer_image_size_reads_both_refined_prompt_and_brief():
    """A vertical hint in the original brief still wires through if Claude drops it."""
    state = make_state(
        service_type="social_graphic",
        brief="9:16 Instagram story please",
        refined_prompt="A neon scene with a person",  # no aspect hint
    )
    assert _infer_image_size(state) == "portrait_16_9"


def test_infer_image_size_handles_missing_fields():
    """Defensive: state without refined_prompt or brief should default to square."""
    state = {"order_id": UUID("00000000-0000-0000-0000-0000000000aa"), "service_type": "social_graphic"}
    assert _infer_image_size(state) == "square_hd"  # type: ignore[arg-type]


def test_portrait_keyword_beats_default_but_not_landscape():
    """If both hints appear, portrait wins (checked first by design)."""
    state = make_state(
        service_type="social_graphic",
        refined_prompt="banner cover photo with vertical reel feel",
    )
    # Portrait list is checked before landscape — first match wins
    assert _infer_image_size(state) == "portrait_16_9"


# ── Base-class integration smoke test ──────────────────────────────────────


@pytest.fixture
def mock_fal() -> MagicMock:
    fal = MagicMock(spec=FalClient)
    fal.endpoint = "fal-ai/flux-pro/v1.1"
    fal.generate = AsyncMock()
    fal.download = AsyncMock(return_value=b"\x89PNGfakejpegbytes")
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


async def test_social_graphics_generator_runs_base_loop_with_inferred_size(
    mock_db, mock_fal, mock_storage, settings
):
    """Smoke test: SocialGraphicsGenerator picks the right size and creates 3 deliverables."""
    mock_db.get_agent_by_key = AsyncMock(
        return_value={
            "id": "00000000-0000-0000-0000-0000000000aa",
            "agent_key": "social_graphics_gen",
            "layer": "generation",
        }
    )
    mock_db.create_deliverable = AsyncMock(
        side_effect=[
            UUID("00000000-0000-0000-0000-00000000d010"),
            UUID("00000000-0000-0000-0000-00000000d011"),
            UUID("00000000-0000-0000-0000-00000000d012"),
        ]
    )

    # Square Instagram feed result (1024x1024)
    mock_fal.generate.return_value = GenerationResult(
        images=tuple(
            GeneratedImage(
                url=f"https://fal.media/sq-{i}.jpg",
                width=1024,
                height=1024,
                content_type="image/jpeg",
            )
            for i in range(3)
        ),
        seed=99,
        cost_usd=0.24,  # 3 x 2 MP x $0.04 for square_hd
        prompt="test",
    )

    agent = SocialGraphicsGenerator(db=mock_db, fal=mock_fal, storage=mock_storage, settings=settings)
    state = make_state(
        service_type="social_graphic",
        refined_prompt="Clean minimalist Instagram feed post for a coffee brand",
        negative_prompt="",
    )
    result = await agent(state)

    # Square preset was passed to fal because no portrait/landscape hints
    gen_kwargs = mock_fal.generate.await_args.kwargs
    assert gen_kwargs["image_size"] == "square_hd"
    assert gen_kwargs["num_images"] == 3

    # 3 deliverables created with the "social-" file prefix
    assert mock_db.create_deliverable.await_count == 3
    file_names = [c.kwargs["file_name"] for c in mock_db.create_deliverable.await_args_list]
    assert all(name.startswith("social-") for name in file_names)

    # State carries the new ids
    assert len(result["deliverable_ids"]) == 3


async def test_social_graphics_routes_portrait_when_story_hinted(
    mock_db, mock_fal, mock_storage, settings
):
    """End-to-end: a story-style brief produces a portrait_16_9 fal call."""
    mock_db.get_agent_by_key = AsyncMock(
        return_value={"id": "00000000-0000-0000-0000-0000000000aa", "agent_key": "social_graphics_gen", "layer": "generation"}
    )
    mock_db.create_deliverable = AsyncMock(return_value=UUID("00000000-0000-0000-0000-00000000d020"))
    mock_fal.generate.return_value = GenerationResult(
        images=(GeneratedImage(url="https://fal.media/v.jpg", width=720, height=1280, content_type="image/jpeg"),),
        seed=1,
        cost_usd=0.04,
        prompt="x",
    )

    agent = SocialGraphicsGenerator(db=mock_db, fal=mock_fal, storage=mock_storage, settings=settings)
    state = make_state(
        service_type="social_graphic",
        refined_prompt="9:16 vertical Instagram story for a fashion drop",
    )
    await agent(state)

    assert mock_fal.generate.await_args.kwargs["image_size"] == "portrait_16_9"


async def test_thumbnail_generator_still_works_after_refactor(
    mock_db, mock_fal, mock_storage, settings
):
    """Regression guard: refactor to GenerationAgentBase didn't break thumbnails."""
    from agency.agents.thumbnail_generator import ThumbnailGenerator

    mock_db.get_agent_by_key = AsyncMock(
        return_value={"id": "00000000-0000-0000-0000-0000000000aa", "agent_key": "thumbnail_gen", "layer": "generation"}
    )
    mock_db.create_deliverable = AsyncMock(return_value=UUID("00000000-0000-0000-0000-00000000d030"))
    mock_fal.generate.return_value = GenerationResult(
        images=(GeneratedImage(url="https://fal.media/t.jpg", width=1280, height=720, content_type="image/jpeg"),),
        seed=2,
        cost_usd=0.04,
        prompt="x",
    )

    agent = ThumbnailGenerator(db=mock_db, fal=mock_fal, storage=mock_storage, settings=settings)
    state = make_state(refined_prompt="cinematic neon thumbnail")
    await agent(state)

    # Still uses landscape_16_9 and "thumbnail-" prefix
    assert mock_fal.generate.await_args.kwargs["image_size"] == "landscape_16_9"
    file_name = mock_db.create_deliverable.await_args.kwargs["file_name"]
    assert file_name.startswith("thumbnail-")
