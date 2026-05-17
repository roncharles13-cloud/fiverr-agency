"""fal.ai client wrapper with cost accounting and image download.

Encapsulates the `fal_client` SDK and adds:
  * Per-call cost computation from image dimensions
  * Image download (fal returns CDN URLs; we re-host in Supabase Storage)
  * Structured `GenerationResult` so callers never touch raw SDK dicts

The SDK reads `FAL_KEY` from the process environment. We set it once when
constructing the client; production deployments should consider only ever
running one `FalClient` per process to avoid env clobbering.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import httpx

# fal_client is imported lazily inside methods so the orchestrator module
# can be imported in environments where fal-client isn't installed
# (e.g. dashboard build environments).
from agency.config import Settings

ImageSize = Literal[
    "square_hd",
    "square",
    "portrait_4_3",
    "portrait_16_9",
    "landscape_4_3",
    "landscape_16_9",
]


@dataclass(frozen=True, slots=True)
class GeneratedImage:
    """One image returned by a generation call. `url` is the fal CDN URL."""

    url: str
    width: int
    height: int
    content_type: str


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Aggregate output of a single generation call."""

    images: tuple[GeneratedImage, ...]
    seed: int | None
    cost_usd: float
    prompt: str


class FalClient:
    """Async wrapper around fal.ai's Flux Pro 1.1 endpoint."""

    # Public pricing for Flux Pro 1.1 as of May 2026:
    # $0.04 per megapixel, rounded UP to the nearest megapixel per image.
    # Source: https://fal.ai/models/fal-ai/flux-pro/v1.1
    USD_PER_MEGAPIXEL: float = 0.04

    # Pixel dimensions for each `image_size` preset, used for cost computation.
    # fal accepts custom dicts too, but presets cover thumbnails / posts / etc.
    SIZE_PRESETS: ClassVar[dict[ImageSize, tuple[int, int]]] = {
        "square_hd": (1024, 1024),
        "square": (512, 512),
        "portrait_4_3": (768, 1024),
        "portrait_16_9": (576, 1024),
        "landscape_4_3": (1024, 768),
        "landscape_16_9": (1024, 576),  # fal Flux Pro 1.1 actual output
    }

    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))

    @classmethod
    def from_settings(cls, settings: Settings) -> FalClient:
        if settings.fal_key is None:
            raise RuntimeError(
                "FAL_KEY is not configured. Set it in .env to use generation agents."
            )
        # The fal_client SDK reads FAL_KEY from the environment. We set it
        # here once per process construction — a single FalClient per process
        # is the assumed deployment model.
        os.environ["FAL_KEY"] = settings.fal_key.get_secret_value()
        return cls(endpoint=settings.fal_flux_pro_endpoint)

    @property
    def endpoint(self) -> str:
        return self._endpoint

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── Generation ──────────────────────────────────────────────────────

    async def generate(
        self,
        *,
        prompt: str,
        image_size: ImageSize = "landscape_16_9",
        num_images: int = 1,
        negative_prompt: str | None = None,
        seed: int | None = None,
        enable_safety_checker: bool = True,
    ) -> GenerationResult:
        """Generate `num_images` images for the given prompt. Returns parsed result + cost."""
        import fal_client  # imported lazily — see module docstring

        arguments: dict[str, Any] = {
            "prompt": prompt,
            "image_size": image_size,
            "num_images": num_images,
            "enable_safety_checker": enable_safety_checker,
        }
        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt
        if seed is not None:
            arguments["seed"] = seed

        raw = await fal_client.subscribe_async(
            self._endpoint, arguments=arguments, with_logs=False
        )
        return _parse_generation_response(raw, image_size=image_size, prompt=prompt)

    # ── Download ────────────────────────────────────────────────────────

    async def download(self, url: str) -> bytes:
        """Fetch an image from a fal CDN URL. Returns raw bytes."""
        resp = await self._http.get(url)
        resp.raise_for_status()
        return resp.content


# ============================================================================
# Pure helpers — testable without hitting fal.ai
# ============================================================================


def _parse_generation_response(
    raw: dict[str, Any],
    *,
    image_size: ImageSize,
    prompt: str,
) -> GenerationResult:
    """Convert the fal SDK's raw response dict into a typed `GenerationResult`."""
    images_raw = raw.get("images") or []
    images = tuple(
        GeneratedImage(
            url=str(img["url"]),
            width=int(img.get("width") or FalClient.SIZE_PRESETS[image_size][0]),
            height=int(img.get("height") or FalClient.SIZE_PRESETS[image_size][1]),
            content_type=str(img.get("content_type") or "image/jpeg"),
        )
        for img in images_raw
    )

    cost = _compute_cost(
        width=images[0].width if images else FalClient.SIZE_PRESETS[image_size][0],
        height=images[0].height if images else FalClient.SIZE_PRESETS[image_size][1],
        n_images=len(images),
    )

    return GenerationResult(
        images=images,
        seed=raw.get("seed"),
        cost_usd=cost,
        prompt=prompt,
    )


def _compute_cost(*, width: int, height: int, n_images: int) -> float:
    """USD cost for `n_images` images at `width x height`.

    fal.ai's Flux Pro 1.1 bills $0.04 per megapixel, rounded UP to the next
    whole megapixel per image. 1280x720 (0.92 MP) bills as 1 MP = $0.04.
    """
    megapixels = math.ceil((width * height) / 1_000_000)
    return megapixels * n_images * FalClient.USD_PER_MEGAPIXEL
