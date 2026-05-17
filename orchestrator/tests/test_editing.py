"""Tests for `agency.editing` pure helpers (Pillow-based)."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from agency.agents.technical_qc import _check
from agency.editing import (
    TextOverlayStyle,
    inspect_image,
    render_text_overlay,
)


def _make_jpeg(width: int, height: int, color: tuple[int, int, int] = (40, 80, 160)) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ── inspect_image ───────────────────────────────────────────────────────────


def test_inspect_image_reads_dimensions_and_format():
    blob = _make_jpeg(1280, 720)
    info = inspect_image(blob)
    assert info.width == 1280
    assert info.height == 720
    assert info.format == "JPEG"
    assert info.size_bytes == len(blob)


def test_inspect_image_raises_on_garbage():
    from PIL import UnidentifiedImageError

    with pytest.raises((UnidentifiedImageError, OSError)):
        inspect_image(b"not an image")


# ── render_text_overlay ─────────────────────────────────────────────────────


def test_render_text_overlay_returns_jpeg_with_same_dimensions():
    src = _make_jpeg(1280, 720)
    out = render_text_overlay(src, "INSANE WIN")

    info = inspect_image(out)
    assert info.format == "JPEG"
    assert info.width == 1280
    assert info.height == 720


def test_render_text_overlay_changes_pixels():
    """Output must differ from input — text was actually composited."""
    src = _make_jpeg(640, 360)
    out = render_text_overlay(src, "HELLO")
    assert out != src


def test_render_text_overlay_position_affects_output():
    src = _make_jpeg(640, 360)
    top = render_text_overlay(src, "X", style=TextOverlayStyle(position="top"))
    bottom = render_text_overlay(src, "X", style=TextOverlayStyle(position="bottom"))
    # Different position → different bytes
    assert top != bottom


# ── _check (Technical QC) ───────────────────────────────────────────────────


def test_check_passes_for_correct_thumbnail():
    blob = _make_jpeg(1280, 720)
    assert _check(blob, expected=(1280, 720)) == []


def test_check_flags_wrong_dimensions():
    blob = _make_jpeg(800, 600)
    failures = _check(blob, expected=(1280, 720))
    assert any("dimensions" in f for f in failures)


def test_check_tolerates_5_percent_drift():
    """1280x720 ±5% should pass; 1280*0.95=1216, so 1220x685 is within tolerance."""
    blob = _make_jpeg(1220, 685)
    assert _check(blob, expected=(1280, 720)) == []


def test_check_flags_unsupported_format():
    # Write a WEBP — disallowed.
    img = Image.new("RGB", (1280, 720), (0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=80)
    failures = _check(buf.getvalue(), expected=(1280, 720))
    assert any("format" in f for f in failures)


def test_check_flags_oversize_file():
    # Build a file claiming to be huge by padding bytes.
    blob = _make_jpeg(1280, 720) + b"\x00" * (11 * 1024 * 1024)
    failures = _check(blob, expected=(1280, 720))
    assert any("too large" in f for f in failures)


def test_check_flags_corrupt_image():
    failures = _check(b"\x00\x01\x02 garbage", expected=(1280, 720))
    assert any("decode failed" in f for f in failures)


def test_check_with_no_expected_skips_dimension_check():
    blob = _make_jpeg(123, 456)
    # Format check still runs; dimension check is skipped
    failures = _check(blob, expected=None)
    assert not any("dimensions" in f for f in failures)
