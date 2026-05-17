"""Unit tests for `IntakeRunner`.

Both `GmailClient` and `IntakeParser` are mocked. The runner's job is purely
orchestration — fetch, parse, label — and the test contract is that:

  * success → `mark_processed`
  * `IntakeExtractionError` → `mark_failed` (terminal, do not retry)
  * any other exception → leave the label alone (transient, retry next cycle)
  * cycle-level exceptions in `run_loop` must not kill the loop
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest

from agency.intake.email_models import FiverrEmail
from agency.intake.gmail_client import GmailClient
from agency.intake.parser import IntakeExtractionError, IntakeParser
from agency.intake.runner import IntakeRunner

ORDER_ID = UUID("00000000-0000-0000-0000-0000000000f0")


def _email(message_id: str) -> FiverrEmail:
    return FiverrEmail(
        message_id=message_id,
        received_at=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
        subject="New order",
        sender="noreply@fiverr.com",
        body_plain="Need a thumbnail for my gaming channel about Minecraft speedruns.",
    )


@pytest.fixture
def mock_gmail() -> MagicMock:
    gmail = MagicMock(spec=GmailClient)
    gmail.list_pending_message_ids = AsyncMock(return_value=[])
    gmail.fetch_email = AsyncMock()
    gmail.mark_processed = AsyncMock()
    gmail.mark_failed = AsyncMock()
    return gmail


@pytest.fixture
def mock_parser() -> MagicMock:
    parser = MagicMock(spec=IntakeParser)
    parser.process_email = AsyncMock(return_value=ORDER_ID)
    return parser


@pytest.fixture
def runner(mock_gmail, mock_parser, settings) -> IntakeRunner:
    return IntakeRunner(gmail=mock_gmail, parser=mock_parser, settings=settings)


# ── Happy path ──────────────────────────────────────────────────────────────


async def test_run_once_processes_each_message(runner, mock_gmail, mock_parser):
    mock_gmail.list_pending_message_ids.return_value = ["m1", "m2", "m3"]
    mock_gmail.fetch_email.side_effect = [_email("m1"), _email("m2"), _email("m3")]

    result = await runner.run_once()

    assert result.processed == 3
    assert result.succeeded == 3
    assert result.failed_terminal == 0
    assert result.failed_transient == 0
    assert result.all_succeeded is True

    assert mock_parser.process_email.await_count == 3
    assert mock_gmail.mark_processed.await_count == 3
    mock_gmail.mark_failed.assert_not_called()


async def test_run_once_with_empty_queue_returns_zero(runner, mock_gmail):
    mock_gmail.list_pending_message_ids.return_value = []
    result = await runner.run_once()
    assert result.processed == 0
    assert result.all_succeeded is False  # nothing processed isn't "success"


# ── Terminal vs transient failure routing ──────────────────────────────────


async def test_extraction_error_marks_email_failed(runner, mock_gmail, mock_parser):
    mock_gmail.list_pending_message_ids.return_value = ["m1"]
    mock_gmail.fetch_email.return_value = _email("m1")
    mock_parser.process_email.side_effect = IntakeExtractionError("bad JSON")

    result = await runner.run_once()

    assert result.processed == 1
    assert result.failed_terminal == 1
    assert result.failed_transient == 0
    mock_gmail.mark_failed.assert_awaited_once_with("m1")
    mock_gmail.mark_processed.assert_not_called()


async def test_transient_error_leaves_email_pending(runner, mock_gmail, mock_parser):
    """Gmail 503 or Supabase blip → leave label alone, retry next cycle."""
    mock_gmail.list_pending_message_ids.return_value = ["m1"]
    mock_gmail.fetch_email.return_value = _email("m1")
    mock_parser.process_email.side_effect = ConnectionError("supabase down")

    result = await runner.run_once()

    assert result.processed == 1
    assert result.failed_transient == 1
    assert result.failed_terminal == 0
    mock_gmail.mark_processed.assert_not_called()
    mock_gmail.mark_failed.assert_not_called()


async def test_mixed_outcomes_in_single_cycle(runner, mock_gmail, mock_parser):
    mock_gmail.list_pending_message_ids.return_value = ["good", "bad", "flaky"]
    mock_gmail.fetch_email.side_effect = [
        _email("good"),
        _email("bad"),
        _email("flaky"),
    ]
    mock_parser.process_email.side_effect = [
        ORDER_ID,                          # good
        IntakeExtractionError("bad JSON"), # bad
        TimeoutError("network blip"),      # flaky
    ]

    result = await runner.run_once()

    assert result.processed == 3
    assert result.succeeded == 1
    assert result.failed_terminal == 1
    assert result.failed_transient == 1
    mock_gmail.mark_processed.assert_awaited_once_with("good")
    mock_gmail.mark_failed.assert_awaited_once_with("bad")


async def test_mark_failed_error_does_not_mask_original_failure(
    runner, mock_gmail, mock_parser
):
    """If labeling fails after a terminal parse error, we still count it as terminal."""
    mock_gmail.list_pending_message_ids.return_value = ["m1"]
    mock_gmail.fetch_email.return_value = _email("m1")
    mock_parser.process_email.side_effect = IntakeExtractionError("bad JSON")
    mock_gmail.mark_failed.side_effect = ConnectionError("gmail down")

    result = await runner.run_once()

    # Outcome is still terminal-failure; the mark_failed error is logged but doesn't
    # change the per-message classification.
    assert result.failed_terminal == 1
    assert result.failed_transient == 0


# ── Loop cancellation ──────────────────────────────────────────────────────


async def test_run_loop_swallows_cycle_errors_and_keeps_going(
    runner, mock_gmail, mock_parser
):
    """A cycle-level exception (not per-email) must not kill the loop."""
    # First cycle blows up before iterating messages; second cycle returns []; we
    # cancel after that so the test terminates.
    call_count = {"n": 0}

    async def _flaky_list(max_results: int) -> list[str]:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("gmail outage")
        # Second call succeeds with empty queue; cancel on third to exit
        if call_count["n"] >= 3:
            raise asyncio.CancelledError
        return []

    mock_gmail.list_pending_message_ids.side_effect = _flaky_list

    with pytest.raises(asyncio.CancelledError):
        await runner.run_loop(interval_seconds=0.0)

    assert call_count["n"] >= 2  # loop survived the first exception
