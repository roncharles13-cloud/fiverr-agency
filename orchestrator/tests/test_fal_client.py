"""Unit tests for the fal.ai client's pure helpers.

The async I/O paths (`generate`, `download`) are exercised end-to-end via
`test_thumbnail_generator.py` with a mocked client. This file covers the
pricing math and response-parsing logic where bugs would silently corrupt
cost reporting or deliverable rows.
"""

from __future__ import annotations

import pytest

from agency.clients.fal_client import (
    FalClient,
    GeneratedImage,
    _compute_cost,
    _parse_generation_response,
)

# ── Cost computation ────────────────────────────────────────────────────────


def test_cost_thumbnail_1280x720_rounds_up_to_1mp():
    # 1280 * 720 = 921_600 px → ceil(0.9216) = 1 MP → $0.04 per image
    cost = _compute_cost(width=1280, height=720, n_images=1)
    assert cost == pytest.approx(0.04)


def test_cost_three_thumbnails():
    cost = _compute_cost(width=1280, height=720, n_images=3)
    assert cost == pytest.approx(0.12)


def test_cost_square_hd_rounds_up_to_2mp():
    # 1024 * 1024 = 1_048_576 px → ceil(1.0486) = 2 MP → $0.08 per image
    cost = _compute_cost(width=1024, height=1024, n_images=1)
    assert cost == pytest.approx(0.08)


def test_cost_fullhd_landscape():
    # 1920 * 1080 = 2_073_600 px → ceil(2.07) = 3 MP → $0.12 per image
    cost = _compute_cost(width=1920, height=1080, n_images=2)
    assert cost == pytest.approx(0.24)


# ── Response parsing ───────────────────────────────────────────────────────


def test_parse_generation_response_typical_shape():
    raw = {
        "images": [
            {"url": "https://fal.media/a.jpg", "width": 1280, "height": 720, "content_type": "image/jpeg"},
            {"url": "https://fal.media/b.jpg", "width": 1280, "height": 720, "content_type": "image/jpeg"},
        ],
        "seed": 42,
        "prompt": "test",
    }
    result = _parse_generation_response(raw, image_size="landscape_16_9", prompt="test")

    assert len(result.images) == 2
    assert all(isinstance(img, GeneratedImage) for img in result.images)
    assert result.images[0].url == "https://fal.media/a.jpg"
    assert result.images[0].width == 1280
    assert result.seed == 42
    assert result.cost_usd == pytest.approx(0.08)  # 2 images * 1 MP * $0.04


def test_parse_generation_response_missing_dimensions_falls_back_to_preset():
    """If fal omits width/height, the response parser uses the preset's nominal size."""
    raw = {
        "images": [
            {"url": "https://fal.media/x.jpg", "content_type": "image/jpeg"},
        ],
        "seed": None,
    }
    result = _parse_generation_response(raw, image_size="square_hd", prompt="test")
    # square_hd preset is 1024x1024
    assert result.images[0].width == 1024
    assert result.images[0].height == 1024
    assert result.cost_usd == pytest.approx(0.08)


def test_parse_generation_response_no_images_costs_zero():
    """Safety filter or empty response — cost is zero, not None."""
    raw = {"images": [], "seed": 1}
    result = _parse_generation_response(raw, image_size="landscape_16_9", prompt="test")
    assert result.images == ()
    assert result.cost_usd == 0.0


def test_size_presets_cover_all_image_size_literal():
    """Every value of the ImageSize Literal must exist in SIZE_PRESETS."""
    # Get the literal options from the type alias
    from typing import get_args

    from agency.clients.fal_client import ImageSize

    for value in get_args(ImageSize):
        assert value in FalClient.SIZE_PRESETS, (
            f"ImageSize literal {value!r} missing from FalClient.SIZE_PRESETS"
        )
