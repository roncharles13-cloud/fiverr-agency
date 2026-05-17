"""Intake runner — orchestrates Gmail fetch → parser → label update.

Distinguishes two failure modes:

  * **Terminal** (`IntakeExtractionError`): the LLM returned output that fails
    `ParsedOrder` validation. Re-fetching the same email and asking again is
    extremely unlikely to succeed — the email is moved to the `Failed` label
    so the operator can review and triage manually.

  * **Transient** (any other exception): Gmail 5xx, network blips, Supabase
    timeouts. The email keeps its `Pending` label and will be retried on the
    next cycle.

`run_loop` swallows cycle-level exceptions and keeps polling. `run_once`
returns a structured summary so the CLI or a test can assert on outcomes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog

from agency.config import Settings
from agency.intake.gmail_client import GmailClient
from agency.intake.parser import IntakeExtractionError, IntakeParser

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class IntakeRunResult:
    """Outcome summary for a single `run_once` invocation."""

    processed: int = 0
    succeeded: int = 0
    failed_terminal: int = 0
    failed_transient: int = 0
    skipped_duplicates: int = 0
    error_summaries: list[str] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return self.processed > 0 and self.failed_terminal == 0 and self.failed_transient == 0


class IntakeRunner:
    """Single-cycle and continuous-loop intake driver."""

    def __init__(
        self,
        gmail: GmailClient,
        parser: IntakeParser,
        settings: Settings,
    ) -> None:
        self._gmail = gmail
        self._parser = parser
        self._settings = settings

    async def run_once(self, max_emails: int | None = None) -> IntakeRunResult:
        """Process the current backlog up to `max_emails` and return a summary."""
        limit = max_emails if max_emails is not None else self._settings.intake_max_per_cycle
        message_ids = await self._gmail.list_pending_message_ids(max_results=limit)

        result = IntakeRunResult()
        print(f"[INTAKE] Processing {len(message_ids)} pending messages", flush=True)
        for mid in message_ids:
            result.processed += 1
            try:
                print(f"[INTAKE] Fetching email {mid}...", flush=True)
                email = await self._gmail.fetch_email(mid)
                print(f"[INTAKE] Parsing email: subject={email.subject!r}, sender={email.sender}", flush=True)
                order_id = await self._parser.process_email(email)
                await self._gmail.mark_processed(mid)
                result.succeeded += 1
                print(f"[INTAKE] SUCCESS: order_id={order_id}", flush=True)
                logger.info(
                    "intake.runner.message_succeeded",
                    message_id=mid,
                    order_id=str(order_id),
                )
            except IntakeExtractionError as exc:
                # Terminal — don't retry forever. Move to Failed label so the
                # operator can inspect.
                result.failed_terminal += 1
                result.error_summaries.append(f"{mid}: extraction failed: {exc}")
                logger.warning(
                    "intake.runner.terminal_failure",
                    message_id=mid,
                    error=str(exc),
                )
                # Best-effort mark — if labeling fails we still surface the original error.
                try:
                    await self._gmail.mark_failed(mid)
                except Exception:
                    logger.exception("intake.runner.mark_failed_errored", message_id=mid)
            except Exception as exc:
                # Transient — leave label as Pending; retry next cycle.
                result.failed_transient += 1
                result.error_summaries.append(
                    f"{mid}: {type(exc).__name__}: {exc}"
                )
                import traceback
                print(f"[INTAKE] TRANSIENT ERROR for {mid}: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                logger.exception(
                    "intake.runner.transient_failure",
                    message_id=mid,
                )
        return result

    async def run_loop(self, interval_seconds: float | None = None) -> None:
        """Poll forever. Cancellation via `KeyboardInterrupt` or `asyncio.CancelledError`."""
        interval = (
            interval_seconds
            if interval_seconds is not None
            else self._settings.intake_poll_interval_seconds
        )
        logger.info("intake.runner.loop_start", interval_seconds=interval)
        while True:
            try:
                result = await self.run_once()
                if result.processed:
                    logger.info(
                        "intake.runner.cycle_summary",
                        processed=result.processed,
                        succeeded=result.succeeded,
                        failed_terminal=result.failed_terminal,
                        failed_transient=result.failed_transient,
                    )
            except asyncio.CancelledError:
                logger.info("intake.runner.loop_cancelled")
                raise
            except Exception as exc:
                import traceback
                print(f"[INTAKE] CYCLE ERROR: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                logger.exception("intake.runner.cycle_errored")
            await asyncio.sleep(interval)
