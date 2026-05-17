"""Supabase client wrapper exposing typed operations the orchestrator needs.

Design:
* Async throughout — LangGraph nodes are async.
* Each method covers exactly one logical table operation. No "smart" queries
  that mix concerns; compose at the call site.
* Returns are plain dicts (validated upstream by pydantic models in the caller
  if needed) to keep this layer thin.
* Service-role key is required. RLS bypass is automatic for service-role.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal
from uuid import UUID

from supabase import AsyncClient, acreate_client

from agency.config import Settings

AgentStateValue = Literal["idle", "processing", "error"]
RunStatusValue = Literal["queued", "processing", "completed", "error", "skipped"]
OrderStatusValue = Literal[
    "pending",
    "clarification_needed",
    "awaiting_response",
    "processing",
    "qc",
    "ready_for_delivery",
    "delivered",
    "error",
    "cancelled",
]


def _iso(value: datetime | None = None) -> str:
    """ISO-8601 timestamp string in UTC, suitable for timestamptz columns."""
    return (value or datetime.now(UTC)).isoformat()


def _drop_none(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip None values so we never overwrite columns with nulls by accident."""
    return {k: v for k, v in payload.items() if v is not None}


class Database:
    """Thin async wrapper around the Supabase Postgres REST client."""

    def __init__(self, client: AsyncClient) -> None:
        self._client = client

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    async def connect(cls, settings: Settings) -> Database:
        """Create an async Supabase client using the service-role key."""
        client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_role_key.get_secret_value(),
        )
        return cls(client)

    @property
    def raw(self) -> AsyncClient:
        """Escape hatch for queries this wrapper does not (yet) cover."""
        return self._client

    # ── Agents registry ─────────────────────────────────────────────────

    async def get_agent_by_key(self, agent_key: str) -> dict[str, Any]:
        result = await (
            self._client.table("agents")
            .select("*")
            .eq("agent_key", agent_key)
            .single()
            .execute()
        )
        if result is None or result.data is None:
            raise LookupError(f"Agent not found: {agent_key}")
        return result.data

    async def list_agents(self) -> list[dict[str, Any]]:
        result = await (
            self._client.table("agents")
            .select("*")
            .order("layer", desc=False)
            .order("layer_order", desc=False)
            .execute()
        )
        return list(result.data or [])

    async def find_generation_agent_for(self, service_type: str) -> dict[str, Any]:
        """Look up the generation agent that handles a given service type."""
        result = await (
            self._client.table("agents")
            .select("*")
            .contains("handles_service_types", [service_type])
            .eq("is_active", True)
            .limit(1)
            .execute()
        )
        rows = result.data or []
        if not rows:
            raise LookupError(f"No generation agent registered for service_type={service_type}")
        return rows[0]

    # ── Orders ──────────────────────────────────────────────────────────

    async def get_order(self, order_id: UUID) -> dict[str, Any] | None:
        result = await (
            self._client.table("orders")
            .select("*")
            .eq("id", str(order_id))
            .maybe_single()
            .execute()
        )
        if result is None:
            return None
        return result.data

    async def find_order_by_idempotency_key(self, key: str) -> dict[str, Any] | None:
        """Look up an existing order by its idempotency key. Returns None if absent."""
        result = await (
            self._client.table("orders")
            .select("*")
            .eq("idempotency_key", key)
            .maybe_single()
            .execute()
        )
        if result is None:
            return None
        return result.data

    async def create_order(
        self,
        *,
        service_type: str,
        brief: str,
        idempotency_key: str | None = None,
        source: str = "fiverr",
        fiverr_order_id: str | None = None,
        client_username: str | None = None,
        client_email: str | None = None,
        reference_images: list[str] | None = None,
        deadline: str | None = None,  # ISO-8601 string
        raw_payload: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Insert a new order row and return its UUID."""
        payload = _drop_none(
            {
                "service_type": service_type,
                "brief": brief,
                "idempotency_key": idempotency_key,
                "source": source,
                "fiverr_order_id": fiverr_order_id,
                "client_username": client_username,
                "client_email": client_email,
                "reference_images": reference_images or [],
                "deadline": deadline,
                "raw_payload": raw_payload or {},
                "metadata": metadata or {},
                "status": "pending",
            }
        )
        result = await self._client.table("orders").insert(payload).execute()
        if not result.data:
            raise RuntimeError("orders insert returned no rows")
        return UUID(result.data[0]["id"])

    async def get_pending_orders(self, limit: int = 10) -> list[dict[str, Any]]:
        result = await (
            self._client.table("orders")
            .select("*")
            .eq("status", "pending")
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return list(result.data or [])

    async def update_order_status(
        self,
        order_id: UUID,
        status: OrderStatusValue,
        confidence_score: float | None = None,
    ) -> None:
        payload = _drop_none({"status": status, "confidence_score": confidence_score})
        await (
            self._client.table("orders")
            .update(payload)
            .eq("id", str(order_id))
            .execute()
        )

    # ── Agent runs ──────────────────────────────────────────────────────

    async def start_agent_run(
        self,
        agent_id: UUID,
        order_id: UUID | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> UUID:
        """Insert a new agent_runs row. `order_id` is nullable for system agents."""
        result = await (
            self._client.table("agent_runs")
            .insert(
                {
                    "agent_id": str(agent_id),
                    "order_id": str(order_id) if order_id else None,
                    "status": "processing",
                    "started_at": _iso(),
                    "input_data": input_data,
                }
            )
            .execute()
        )
        if not result.data:
            raise RuntimeError("agent_runs insert returned no rows")
        return UUID(result.data[0]["id"])

    async def finish_agent_run(
        self,
        run_id: UUID,
        status: RunStatusValue,
        log_summary: str | None = None,
        input_data: dict[str, Any] | None = None,
        output_data: dict[str, Any] | None = None,
        error_message: str | None = None,
        cost_usd: float | None = None,
        order_id: UUID | None = None,
    ) -> None:
        """Terminal update for an agent_runs row.

        `order_id` is only included in the update if non-None — passing `None`
        leaves the existing value untouched (it might already be set from
        `start_agent_run`, or it might legitimately stay NULL for a failed
        intake before the order was created).
        """
        payload = _drop_none(
            {
                "status": status,
                "completed_at": _iso(),
                "log_summary": log_summary,
                "input_data": input_data,
                "output_data": output_data,
                "error_message": error_message,
                "cost_usd": cost_usd,
                "order_id": str(order_id) if order_id else None,
            }
        )
        await (
            self._client.table("agent_runs")
            .update(payload)
            .eq("id", str(run_id))
            .execute()
        )

    # ── Agent status (live snapshot) ────────────────────────────────────
    # `total_runs` / `total_errors` are maintained by the DB trigger
    # `agent_runs_update_counters` — do not increment them from here.

    async def set_agent_status(
        self,
        agent_id: UUID,
        current_status: AgentStateValue,
        current_order_id: UUID | None = None,
        current_run_id: UUID | None = None,
        last_log: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "current_status": current_status,
            "current_order_id": str(current_order_id) if current_order_id else None,
            "current_run_id": str(current_run_id) if current_run_id else None,
        }
        if last_log is not None:
            payload["last_log"] = last_log
        await (
            self._client.table("agent_status")
            .update(payload)
            .eq("agent_id", str(agent_id))
            .execute()
        )

    # ── Deliverables ────────────────────────────────────────────────────

    async def list_deliverables(
        self,
        order_id: UUID,
        *,
        only_approved: bool = False,
    ) -> list[dict[str, Any]]:
        """All deliverables for an order, ordered by variant_index then creation."""
        q = (
            self._client.table("deliverables")
            .select("*")
            .eq("order_id", str(order_id))
            .order("variant_index")
            .order("created_at")
        )
        if only_approved:
            q = q.eq("is_approved", True)
        result = await q.execute()
        return list(result.data or [])

    async def update_deliverable_qc(
        self,
        deliverable_id: UUID,
        *,
        technical_qc_passed: bool | None = None,
        quality_score: float | None = None,
        brand_consistency_score: float | None = None,
        is_approved: bool | None = None,
    ) -> None:
        payload = _drop_none(
            {
                "technical_qc_passed": technical_qc_passed,
                "quality_score": quality_score,
                "brand_consistency_score": brand_consistency_score,
                "is_approved": is_approved,
            }
        )
        if not payload:
            return
        await (
            self._client.table("deliverables")
            .update(payload)
            .eq("id", str(deliverable_id))
            .execute()
        )

    async def create_delivery_package(
        self,
        *,
        order_id: UUID,
        delivery_message: str,
        zip_url: str | None = None,
        upsell_suggestion: str | None = None,
    ) -> UUID:
        """Insert a delivery_packages row in pending_approval status."""
        payload = _drop_none(
            {
                "order_id": str(order_id),
                "delivery_message": delivery_message,
                "zip_url": zip_url,
                "upsell_suggestion": upsell_suggestion,
            }
        )
        result = await self._client.table("delivery_packages").insert(payload).execute()
        if not result.data:
            raise RuntimeError("delivery_packages insert returned no rows")
        return UUID(result.data[0]["id"])

    async def create_deliverable(
        self,
        *,
        order_id: UUID,
        file_url: str,
        file_name: str,
        file_type: str,
        produced_by_agent_id: UUID | None = None,
        produced_by_run_id: UUID | None = None,
        file_size_bytes: int | None = None,
        dimensions: dict[str, Any] | None = None,
        variant_index: int = 0,
        parent_deliverable_id: UUID | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> UUID:
        """Insert a deliverable row and return its UUID."""
        payload = _drop_none(
            {
                "order_id": str(order_id),
                "produced_by_agent_id": str(produced_by_agent_id) if produced_by_agent_id else None,
                "produced_by_run_id": str(produced_by_run_id) if produced_by_run_id else None,
                "parent_deliverable_id": (
                    str(parent_deliverable_id) if parent_deliverable_id else None
                ),
                "file_url": file_url,
                "file_name": file_name,
                "file_type": file_type,
                "file_size_bytes": file_size_bytes,
                "dimensions": dimensions,
                "variant_index": variant_index,
                "metadata": metadata or {},
            }
        )
        result = await self._client.table("deliverables").insert(payload).execute()
        if not result.data:
            raise RuntimeError("deliverables insert returned no rows")
        return UUID(result.data[0]["id"])

    # ── Clarification requests ──────────────────────────────────────────

    async def create_clarification_request(
        self,
        order_id: UUID,
        questions: list[str],
        draft_message: str,
    ) -> UUID:
        result = await (
            self._client.table("clarification_requests")
            .insert(
                {
                    "order_id": str(order_id),
                    "questions": questions,
                    "draft_message": draft_message,
                    "status": "drafted",
                }
            )
            .execute()
        )
        if not result.data:
            raise RuntimeError("clarification_requests insert returned no rows")
        return UUID(result.data[0]["id"])
