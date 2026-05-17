"""Unit tests for the Prompt Engineering agent.

Mocks the Anthropic client. Verifies:
  * Service-type-specific system prompt selection
  * Refined prompt and negative prompt propagated to state
  * Text overlay carried separately (not inlined into the generation prompt)
  * Empty refined_prompt is rejected (defensive)
  * Cost recorded on the run
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from agency.agents.prompt_engineering import (
    _SYSTEM_PROMPTS,
    PromptEngineering,
    _build_user_prompt,
)
from tests.conftest import make_state


@pytest.fixture
def prompt_engineering(mock_db, mock_anthropic) -> PromptEngineering:
    mock_db.get_agent_by_key = AsyncMock(
        return_value={
            "id": "00000000-0000-0000-0000-0000000000aa",
            "agent_key": "prompt_engineering",
            "display_name": "Prompt Engineering",
            "layer": "creative",
        }
    )
    return PromptEngineering(db=mock_db, anthropic=mock_anthropic)


async def test_happy_path_thumbnail(prompt_engineering, mock_anthropic, mock_db, completion_result_factory):
    mock_anthropic.complete_json.return_value = (
        {
            "refined_prompt": (
                "Cinematic 16:9 YouTube thumbnail, shocked young gamer face on left, "
                "neon explosion behind, rim-lit, high contrast saturated colors"
            ),
            "negative_prompt": "blurry, washed-out, ugly anatomy, watermark",
            "text_overlay": "INSANE WIN",
            "rationale": "Lead with subject, layer mood; text deferred to overlay agent.",
        },
        completion_result_factory(cost_usd=0.003),
    )

    state = make_state(service_type="thumbnail", brief="MrBeast-style gaming thumbnail with INSANE WIN title.")
    result = await prompt_engineering(state)

    assert "Cinematic" in result["refined_prompt"]
    assert result["negative_prompt"] == "blurry, washed-out, ugly anatomy, watermark"
    assert result["text_overlay"] == "INSANE WIN"

    # Cost was recorded
    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    assert finish_kwargs["cost_usd"] == pytest.approx(0.003)


async def test_service_type_selects_correct_system_prompt(prompt_engineering, mock_anthropic, completion_result_factory):
    """Different service types must use different system prompts."""
    mock_anthropic.complete_json.return_value = (
        {
            "refined_prompt": "Professional headshot, 85mm portrait lens, soft window light, neutral backdrop",
            "negative_prompt": "plastic skin, overexposed",
            "text_overlay": None,
            "rationale": "Specified lens + lighting per headshot rubric.",
        },
        completion_result_factory(),
    )

    state = make_state(service_type="headshot", brief="Professional LinkedIn-ready headshot for a finance executive.")
    await prompt_engineering(state)

    # The system prompt argument should have been the headshot one
    call_kwargs = mock_anthropic.complete_json.await_args.kwargs
    system_used = call_kwargs["system"]
    assert system_used is _SYSTEM_PROMPTS["headshot"]
    assert system_used is not _SYSTEM_PROMPTS["thumbnail"]


async def test_empty_refined_prompt_is_rejected(prompt_engineering, mock_anthropic, completion_result_factory):
    mock_anthropic.complete_json.return_value = (
        {
            "refined_prompt": "",  # invalid — must reject
            "negative_prompt": "",
            "text_overlay": None,
            "rationale": "",
        },
        completion_result_factory(),
    )

    state = make_state()
    with pytest.raises(ValueError, match="empty refined_prompt"):
        await prompt_engineering(state)


async def test_unknown_service_type_raises():
    """Defensive: a service_type the agent has no system prompt for must error."""
    from unittest.mock import MagicMock
    agent = PromptEngineering(db=MagicMock(), anthropic=MagicMock())
    state = {"service_type": "video", "brief": "x", "order_id": None, "reference_image_urls": []}
    # Bypass the lifecycle wrapper by calling execute directly with a stub run
    run = MagicMock()
    with pytest.raises(ValueError, match="No system prompt"):
        await agent.execute(state, run)  # type: ignore[arg-type]


def test_user_prompt_includes_style_when_present():
    """If state.style_attributes is set, it must appear in the user prompt."""
    prompt = _build_user_prompt(
        brief="A modern thumbnail",
        service_type="thumbnail",
        style={"palette": ["#FF0000", "#00FF00"], "mood": "energetic"},
    )
    assert "Style reference" in prompt
    assert "palette" in prompt
    assert "energetic" in prompt


def test_user_prompt_omits_style_section_when_absent():
    prompt = _build_user_prompt(brief="A modern thumbnail", service_type="thumbnail", style=None)
    assert "Style reference" not in prompt
