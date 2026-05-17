"""Orchestrator CLI.

Commands:
  * `agency auth-gmail`    — one-time interactive OAuth flow; writes `token.json`
  * `agency intake-once`   — fetch + parse pending Fiverr emails, then exit
  * `agency intake-loop`   — poll Gmail forever (Ctrl+C to stop)
  * `agency run-once`      — pick one pending order and run it through the graph
  * `agency run-loop`      — TODO (not yet implemented)

Runtime construction is split so that intake commands don't need to build the
LangGraph state machine (and vice versa). This keeps `auth-gmail` runnable
even before Supabase is configured.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from typing import Any
from uuid import UUID

import structlog

from agency.agents.background_removal_agent import BackgroundRemovalAgent
from agency.agents.brief_clarification import BriefClarification
from agency.agents.business_design_generator import BusinessDesignGenerator
from agency.agents.delivery_packager import DeliveryPackager
from agency.agents.headshot_generator import HeadshotGenerator
from agency.agents.logo_generator import LogoGenerator
from agency.agents.prompt_engineering import PromptEngineering
from agency.agents.social_graphics_generator import SocialGraphicsGenerator
from agency.agents.technical_qc import TechnicalQC
from agency.agents.text_renderer import TextRenderer
from agency.agents.thumbnail_generator import ThumbnailGenerator
from agency.agents.visual_qc import VisualQC
from agency.clients.anthropic_client import AnthropicClient
from agency.clients.fal_client import FalClient
from agency.config import Settings, get_settings
from agency.db import Database
from agency.graph import build_graph
from agency.intake.gmail_client import GmailClient
from agency.intake.parser import IntakeParser
from agency.intake.runner import IntakeRunner
from agency.state import WorkflowState
from agency.storage.supabase_storage import SupabaseStorage

logger = structlog.get_logger(__name__)


# ============================================================================
# Runtimes — one per command family so we only build what we need.
# ============================================================================


@dataclass(slots=True)
class GraphRuntime:
    """Long-lived clients + compiled graph for `run-once` / `run-loop`."""

    settings: Settings
    db: Database
    anthropic: AnthropicClient
    fal: FalClient
    storage: SupabaseStorage
    graph: Any  # CompiledStateGraph — typed Any to keep import surface light

    async def aclose(self) -> None:
        await self.anthropic.aclose()
        await self.fal.aclose()


@dataclass(slots=True)
class IntakeRuntime:
    """Long-lived clients + runner for `intake-once` / `intake-loop`."""

    settings: Settings
    db: Database
    anthropic: AnthropicClient
    gmail: GmailClient
    runner: IntakeRunner

    async def aclose(self) -> None:
        await self.anthropic.aclose()


async def _build_graph_runtime() -> GraphRuntime:
    settings = get_settings()
    db = await Database.connect(settings)
    anthropic = AnthropicClient.from_settings(settings)
    fal = FalClient.from_settings(settings)
    storage = SupabaseStorage(client=db.raw)

    graph = build_graph(
        brief_clarification=BriefClarification(db=db, anthropic=anthropic, settings=settings),
        prompt_engineering=PromptEngineering(db=db, anthropic=anthropic),
        thumbnail_gen=ThumbnailGenerator(db=db, fal=fal, storage=storage, settings=settings),
        social_graphics_gen=SocialGraphicsGenerator(db=db, fal=fal, storage=storage, settings=settings),
        headshot_gen=HeadshotGenerator(db=db, fal=fal, storage=storage, settings=settings),
        business_design_gen=BusinessDesignGenerator(db=db, fal=fal, storage=storage, settings=settings),
        logo_gen=LogoGenerator(db=db, fal=fal, storage=storage, settings=settings),
        background_removal=BackgroundRemovalAgent(db=db, storage=storage, settings=settings),
        text_renderer=TextRenderer(db=db, storage=storage, settings=settings),
        technical_qc=TechnicalQC(db=db, storage=storage, settings=settings),
        visual_qc=VisualQC(db=db, anthropic=anthropic, storage=storage, settings=settings),
        delivery_packager=DeliveryPackager(db=db, storage=storage, settings=settings),
    )
    return GraphRuntime(
        settings=settings, db=db, anthropic=anthropic, fal=fal, storage=storage, graph=graph
    )


async def _build_intake_runtime() -> IntakeRuntime:
    settings = get_settings()
    db = await Database.connect(settings)
    anthropic = AnthropicClient.from_settings(settings)
    gmail = await GmailClient.connect(settings)
    parser = IntakeParser(db=db, anthropic=anthropic)
    runner = IntakeRunner(gmail=gmail, parser=parser, settings=settings)
    return IntakeRuntime(
        settings=settings, db=db, anthropic=anthropic, gmail=gmail, runner=runner
    )


# ============================================================================
# Command implementations
# ============================================================================


def _initial_state(order: dict[str, Any]) -> WorkflowState:
    """Build the LangGraph initial state from a Supabase order row."""
    refs = order.get("reference_images") or []
    return WorkflowState(
        order_id=UUID(order["id"]),
        service_type=order["service_type"],
        brief=order["brief"],
        reference_image_urls=list(refs) if isinstance(refs, list) else [],
    )


async def _run_once() -> int:
    runtime = await _build_graph_runtime()
    try:
        orders = await runtime.db.get_pending_orders(limit=1)
        if not orders:
            logger.info("run_once.no_pending_orders")
            return 0

        order = orders[0]
        logger.info("run_once.processing", order_id=order["id"])

        initial = _initial_state(order)
        final = await runtime.graph.ainvoke(initial)

        logger.info(
            "run_once.complete",
            order_id=order["id"],
            confidence_score=final.get("confidence_score"),
            clarification_needed=final.get("clarification_needed"),
        )
        return 0
    finally:
        await runtime.aclose()


async def _run_loop(interval: float | None) -> int:
    """Poll for pending orders and process each through the graph.

    Two layers of error containment:
      * Per-order: an exception in the graph is logged and the loop continues
        (the order's `agent_runs` row carries the failure details).
      * Per-cycle: a list-orders failure (Supabase blip) is logged and we sleep
        before retrying.
    """
    runtime = await _build_graph_runtime()
    poll_interval = interval if interval is not None else 30.0
    logger.info("run_loop.start", interval_seconds=poll_interval)

    try:
        while True:
            try:
                orders = await runtime.db.get_pending_orders(limit=1)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("run_loop.list_orders_failed")
                await asyncio.sleep(poll_interval)
                continue

            if not orders:
                await asyncio.sleep(poll_interval)
                continue

            order = orders[0]
            order_id = order["id"]
            try:
                logger.info("run_loop.processing", order_id=order_id)
                initial = _initial_state(order)
                final = await runtime.graph.ainvoke(initial)
                logger.info(
                    "run_loop.complete",
                    order_id=order_id,
                    clarification_needed=final.get("clarification_needed"),
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("run_loop.order_failed", order_id=order_id)
                # Mark the order errored so it's not picked up again next cycle.
                try:
                    await runtime.db.update_order_status(
                        order_id=UUID(order_id), status="error"
                    )
                except Exception:
                    logger.exception("run_loop.mark_error_failed", order_id=order_id)
            # No sleep between orders — drain the queue greedily.
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("run_loop.cancelled")
        return 0
    finally:
        await runtime.aclose()


async def _intake_once(max_emails: int | None) -> int:
    runtime = await _build_intake_runtime()
    try:
        result = await runtime.runner.run_once(max_emails=max_emails)
        logger.info(
            "intake_once.complete",
            processed=result.processed,
            succeeded=result.succeeded,
            failed_terminal=result.failed_terminal,
            failed_transient=result.failed_transient,
        )
        # Exit code 1 if any email failed terminally — useful for CI / cron alerting.
        return 1 if result.failed_terminal > 0 else 0
    finally:
        await runtime.aclose()


async def _intake_loop(interval: float | None) -> int:
    runtime = await _build_intake_runtime()
    try:
        await runtime.runner.run_loop(interval_seconds=interval)
        return 0  # unreachable except via cancellation
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("intake_loop.cancelled")
        return 0
    finally:
        await runtime.aclose()


async def _auth_gmail() -> int:
    settings = get_settings()
    await GmailClient.authorize_interactive(settings)
    logger.info("auth_gmail.success", token_file=settings.gmail_token_file)
    return 0


# ============================================================================
# Logging + argv plumbing
# ============================================================================


def _configure_logging(level: str) -> None:
    logging.basicConfig(format="%(message)s", level=level.upper())
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        cache_logger_on_first_use=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agency", description="Fiverr AI Agency orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("auth-gmail", help="One-time OAuth flow; produces token.json")

    p_intake_once = sub.add_parser("intake-once", help="Fetch + parse pending Fiverr emails")
    p_intake_once.add_argument("--max-emails", type=int, default=None)

    p_intake_loop = sub.add_parser("intake-loop", help="Poll Gmail for Fiverr emails forever")
    p_intake_loop.add_argument(
        "--interval", type=float, default=None, help="Poll interval in seconds"
    )

    sub.add_parser("run-once", help="Process one pending order through the graph")

    p_run_loop = sub.add_parser("run-loop", help="Poll for pending orders forever and run them through the graph")
    p_run_loop.add_argument(
        "--interval", type=float, default=None, help="Poll interval (seconds) when queue is empty"
    )

    args = parser.parse_args(argv)

    settings = get_settings()
    _configure_logging(settings.log_level)

    if args.command == "auth-gmail":
        return asyncio.run(_auth_gmail())
    if args.command == "intake-once":
        return asyncio.run(_intake_once(args.max_emails))
    if args.command == "intake-loop":
        return asyncio.run(_intake_loop(args.interval))
    if args.command == "run-once":
        return asyncio.run(_run_once())
    if args.command == "run-loop":
        return asyncio.run(_run_loop(args.interval))

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
