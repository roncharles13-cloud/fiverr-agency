"""Agent lifecycle wrapper.

Every agent in the pipeline writes status the same way (insert agent_runs row,
flip agent_status to processing, then update both on success or error). This
module hides that mechanics so individual agents only carry business logic.

Usage:

    async with agent_lifecycle(db, "thumbnail_gen", order_id) as run:
        run.set_input(brief=brief)
        deliverables = await do_work()
        run.set_output(deliverable_ids=[str(d) for d in deliverables])
        run.log("Generated 3 variations")
        run.add_cost(0.12)

On clean exit: marks the run completed and the agent idle.
On exception: marks the run errored and the agent errored, then re-raises so
LangGraph can decide whether to retry.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID

import structlog

from agency.db import Database

logger = structlog.get_logger(__name__)


class AgentRun:
    """Mutable handle returned by `agent_lifecycle`. Pass to the agent body."""

    __slots__ = (
        "agent_id",
        "cost_usd",
        "input_data",
        "log_summary",
        "order_id",
        "output_data",
        "run_id",
    )

    def __init__(self, run_id: UUID, agent_id: UUID, order_id: UUID | None) -> None:
        self.run_id = run_id
        self.agent_id = agent_id
        self.order_id: UUID | None = order_id
        self.input_data: dict[str, Any] | None = None
        self.output_data: dict[str, Any] | None = None
        self.log_summary: str | None = None
        self.cost_usd: float | None = None

    def set_input(self, **kwargs: Any) -> None:
        """Record the agent's inputs for the audit trail."""
        self.input_data = {**(self.input_data or {}), **kwargs}

    def set_output(self, **kwargs: Any) -> None:
        """Record the agent's outputs for the audit trail."""
        self.output_data = {**(self.output_data or {}), **kwargs}

    def log(self, summary: str) -> None:
        """Set the one-line summary shown on the dashboard for this run."""
        self.log_summary = summary

    def add_cost(self, usd: float) -> None:
        """Accumulate API cost incurred during this run."""
        self.cost_usd = (self.cost_usd or 0.0) + usd

    def set_order_id(self, order_id: UUID) -> None:
        """Attach an order_id discovered or created during execution.

        Only meaningful for "system" agents (Intake Parser) that run before the
        order row exists. The lifecycle wrapper will update `agent_runs.order_id`
        on exit if this was called.
        """
        self.order_id = order_id


@asynccontextmanager
async def agent_lifecycle(
    db: Database,
    agent_key: str,
    order_id: UUID | None = None,
) -> AsyncIterator[AgentRun]:
    """Manage the full execution lifecycle of one agent run.

    Yields an `AgentRun` that the caller populates. On clean exit the run is
    marked completed; on exception it is marked errored and the exception is
    re-raised.

    `order_id` may be `None` for system agents (Intake Parser) that create the
    order during execution. In that case the caller must invoke
    `run.set_order_id(...)` once the order exists; the lifecycle attaches it to
    `agent_runs.order_id` on exit. If never set, the run row keeps `order_id = NULL`.
    """
    agent = await db.get_agent_by_key(agent_key)
    agent_id = UUID(agent["id"])

    run_id = await db.start_agent_run(agent_id=agent_id, order_id=order_id)
    await db.set_agent_status(
        agent_id=agent_id,
        current_status="processing",
        current_order_id=order_id,
        current_run_id=run_id,
        last_log=None,
    )
    logger.info(
        "agent.started",
        agent_key=agent_key,
        run_id=str(run_id),
        order_id=str(order_id) if order_id else None,
    )

    run = AgentRun(run_id=run_id, agent_id=agent_id, order_id=order_id)

    try:
        yield run
    except Exception as exc:
        await db.finish_agent_run(
            run_id=run_id,
            status="error",
            log_summary=run.log_summary,
            input_data=run.input_data,
            output_data=run.output_data,
            error_message=f"{type(exc).__name__}: {exc}",
            cost_usd=run.cost_usd,
            order_id=run.order_id,  # may have been set mid-execution
        )
        await db.set_agent_status(
            agent_id=agent_id,
            current_status="error",
            current_order_id=run.order_id,
            current_run_id=run_id,
            last_log=run.log_summary or f"error: {exc}",
        )
        logger.exception(
            "agent.failed",
            agent_key=agent_key,
            run_id=str(run_id),
            order_id=str(run.order_id) if run.order_id else None,
        )
        raise
    else:
        await db.finish_agent_run(
            run_id=run_id,
            status="completed",
            log_summary=run.log_summary,
            input_data=run.input_data,
            output_data=run.output_data,
            cost_usd=run.cost_usd,
            order_id=run.order_id,
        )
        await db.set_agent_status(
            agent_id=agent_id,
            current_status="idle",
            current_order_id=None,
            current_run_id=None,
            last_log=run.log_summary,
        )
        logger.info(
            "agent.completed",
            agent_key=agent_key,
            run_id=str(run_id),
            order_id=str(run.order_id) if run.order_id else None,
            cost_usd=run.cost_usd,
        )
