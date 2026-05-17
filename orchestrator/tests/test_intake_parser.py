"""Unit tests for the Intake Parser.

Mocks the Anthropic client and the Database. Verifies:
  * Happy path: Claude returns valid JSON, order is inserted, run records cost.
  * Idempotency: a duplicate `message_id` skips re-parsing and reuses the
    existing order.
  * Validation: invalid JSON or missing required fields raises
    `IntakeExtractionError` and the run is recorded as errored.
  * Order_id attachment: the new order_id is propagated to the lifecycle.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID

import pytest

from agency.intake.email_models import FiverrEmail
from agency.intake.parser import IntakeExtractionError, IntakeParser

NEW_ORDER_ID = UUID("00000000-0000-0000-0000-0000000000e0")
EXISTING_ORDER_ID = UUID("00000000-0000-0000-0000-0000000000e1")
EXISTING_MESSAGE_ID = "<existing@gmail>"


@pytest.fixture
def email() -> FiverrEmail:
    return FiverrEmail(
        message_id="<msg-abc-123@mail.gmail.com>",
        received_at=datetime(2026, 5, 14, 12, 0, tzinfo=UTC),
        subject="New order: YouTube thumbnail for gaming channel",
        sender="noreply@fiverr.com",
        body_plain=(
            "You received a new order from @pixelrush.\n\n"
            "Service: I will design high-CTR YouTube thumbnails for gaming, finance, and lifestyle channels\n"
            "Order details: I need a thumbnail for my new MrBeast-style gaming video. "
            "Big shocked face on the left, neon explosion behind, title 'INSANE WIN' in bold yellow. "
            "1280x720, very high contrast.\n\n"
            "Deadline: 2026-05-16T18:00:00Z"
        ),
        attachment_urls=[],
    )


@pytest.fixture
def intake_parser(mock_db, mock_anthropic) -> IntakeParser:
    # Override the mock_db.get_agent_by_key default to return the intake_parser agent row.
    mock_db.get_agent_by_key = AsyncMock(
        return_value={
            "id": "00000000-0000-0000-0000-0000000000aa",
            "agent_key": "intake_parser",
            "display_name": "Intake Parser",
            "layer": "coordination",
        }
    )
    mock_db.create_order = AsyncMock(return_value=NEW_ORDER_ID)
    mock_db.find_order_by_idempotency_key = AsyncMock(return_value=None)
    return IntakeParser(db=mock_db, anthropic=mock_anthropic)


async def test_happy_path_parses_and_inserts(
    intake_parser, mock_db, mock_anthropic, email, completion_result_factory
):
    mock_anthropic.complete_json.return_value = (
        {
            "service_type": "thumbnail",
            "brief": (
                "MrBeast-style gaming thumbnail. Big shocked face on left, neon explosion "
                "behind, title 'INSANE WIN' in bold yellow. 1280x720, high contrast."
            ),
            "fiverr_order_id": None,
            "client_username": "pixelrush",
            "deadline": "2026-05-16T18:00:00+00:00",
            "reference_image_urls": [],
            "confidence": 0.95,
            "notes": None,
        },
        completion_result_factory(cost_usd=0.0023),
    )

    order_id = await intake_parser.process_email(email)

    assert order_id == NEW_ORDER_ID

    # The idempotency key was derived from the message_id and used both for the
    # duplicate check and the insert.
    mock_db.find_order_by_idempotency_key.assert_awaited_once_with(
        f"gmail:{email.message_id}"
    )
    mock_db.create_order.assert_awaited_once()
    create_kwargs = mock_db.create_order.await_args.kwargs
    assert create_kwargs["idempotency_key"] == f"gmail:{email.message_id}"
    assert create_kwargs["service_type"] == "thumbnail"
    assert create_kwargs["client_username"] == "pixelrush"
    assert create_kwargs["client_email"] == email.sender
    assert create_kwargs["deadline"] == "2026-05-16T18:00:00+00:00"
    assert "intake_confidence" in create_kwargs["metadata"]
    assert create_kwargs["raw_payload"]["message_id"] == email.message_id

    # Cost was recorded on the run and the new order_id was attached
    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    assert finish_kwargs["status"] == "completed"
    assert finish_kwargs["cost_usd"] == pytest.approx(0.0023)
    assert finish_kwargs["order_id"] == NEW_ORDER_ID


async def test_duplicate_message_returns_existing_order_without_parsing(
    intake_parser, mock_db, mock_anthropic, email
):
    mock_db.find_order_by_idempotency_key = AsyncMock(
        return_value={"id": str(EXISTING_ORDER_ID), "service_type": "thumbnail"}
    )

    returned = await intake_parser.process_email(email)

    assert returned == EXISTING_ORDER_ID
    # Neither Claude nor the order-insert path was touched.
    mock_anthropic.complete_json.assert_not_called()
    mock_db.create_order.assert_not_called()
    # No agent_run was started for a duplicate — duplicate detection is pre-lifecycle.
    mock_db.start_agent_run.assert_not_called()


async def test_invalid_extraction_raises_and_records_error(
    intake_parser, mock_db, mock_anthropic, email, completion_result_factory
):
    # Brief is too short → pydantic validator rejects it.
    mock_anthropic.complete_json.return_value = (
        {
            "service_type": "thumbnail",
            "brief": "make logo",
            "fiverr_order_id": None,
            "client_username": "anon",
            "deadline": None,
            "reference_image_urls": [],
            "confidence": 0.4,
            "notes": "ambiguous",
        },
        completion_result_factory(cost_usd=0.001),
    )

    with pytest.raises(IntakeExtractionError):
        await intake_parser.process_email(email)

    # Order was NOT created
    mock_db.create_order.assert_not_called()

    # Lifecycle recorded the error
    finish_kwargs = mock_db.finish_agent_run.await_args.kwargs
    assert finish_kwargs["status"] == "error"
    assert "IntakeExtractionError" in finish_kwargs["error_message"]


async def test_invalid_service_type_raises(
    intake_parser, mock_db, mock_anthropic, email, completion_result_factory
):
    """The model invents a service type that isn't in our enum — must reject."""
    mock_anthropic.complete_json.return_value = (
        {
            "service_type": "video_edit",  # not a valid ServiceType
            "brief": "Edit a 10-second video clip with motion graphics overlay.",
            "fiverr_order_id": None,
            "client_username": "anon",
            "deadline": None,
            "reference_image_urls": [],
            "confidence": 0.8,
            "notes": None,
        },
        completion_result_factory(),
    )

    with pytest.raises(IntakeExtractionError):
        await intake_parser.process_email(email)

    mock_db.create_order.assert_not_called()


async def test_falls_back_to_email_attachments_when_extraction_omits_references(
    intake_parser, mock_db, mock_anthropic, email, completion_result_factory
):
    """If Claude doesn't surface reference urls but the email had attachments, use them."""
    email_with_attachments = email.model_copy(
        update={
            "attachment_urls": [
                "https://fiverr.example/ref1.png",
                "https://fiverr.example/ref2.jpg",
            ]
        }
    )

    mock_anthropic.complete_json.return_value = (
        {
            "service_type": "thumbnail",
            "brief": "Reference-driven thumbnail in the style of attached images.",
            "fiverr_order_id": None,
            "client_username": "pixelrush",
            "deadline": None,
            "reference_image_urls": [],  # Claude didn't extract them
            "confidence": 0.85,
            "notes": None,
        },
        completion_result_factory(),
    )

    await intake_parser.process_email(email_with_attachments)

    create_kwargs = mock_db.create_order.await_args.kwargs
    assert create_kwargs["reference_images"] == email_with_attachments.attachment_urls
