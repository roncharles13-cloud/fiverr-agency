"""Abstract base class for pipeline agents.

Each concrete agent subclasses `Agent`, sets `agent_key` to match a row in the
`agents` table, and implements `execute(state, run)` with business logic. The
`__call__` method is the LangGraph node entry point — it wraps `execute` in
the agent lifecycle context manager so the audit trail and live status updates
are uniform across the entire pipeline.

Convention: `execute` returns a *partial* state dict containing only the keys
this agent produces. LangGraph merges that into the running `WorkflowState`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from agency.db import Database
from agency.lifecycle import AgentRun, agent_lifecycle
from agency.state import WorkflowState


class Agent(ABC):
    """Base class every pipeline agent inherits from."""

    #: Must match the `agent_key` column of one row in `public.agents`.
    agent_key: ClassVar[str]

    def __init__(self, db: Database) -> None:
        self.db = db

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        # Only enforce agent_key on concrete classes that explicitly declare it.
        # Abstract intermediate bases (e.g. GenerationAgentBase) omit agent_key
        # from their own __dict__ and are skipped here.
        if "agent_key" in cls.__dict__:
            if not isinstance(cls.agent_key, str) or not cls.agent_key:
                raise TypeError(
                    f"{cls.__name__} must declare a non-empty `agent_key` class variable "
                    f"matching a row in the `agents` table."
                )

    async def __call__(self, state: WorkflowState) -> WorkflowState:
        """LangGraph node entry point.

        Wraps `execute` with the lifecycle context manager. Returns the merged
        state so LangGraph treats this as a node update.
        """
        async with agent_lifecycle(
            db=self.db,
            agent_key=self.agent_key,
            order_id=state["order_id"],
        ) as run:
            partial = await self.execute(state, run)
        return {**state, **partial}

    @abstractmethod
    async def execute(self, state: WorkflowState, run: AgentRun) -> WorkflowState:
        """Implement business logic here.

        Return a partial `WorkflowState` containing only the keys this agent
        produces. Use `run.set_input(...)`, `run.set_output(...)`, `run.log(...)`
        and `run.add_cost(...)` to populate the audit trail.

        Raise on unrecoverable error — the lifecycle wrapper records the
        exception and re-raises so the orchestrator can decide whether to retry.
        """
        raise NotImplementedError
