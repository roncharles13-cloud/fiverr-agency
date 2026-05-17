"""Gmail API wrapper for the intake pipeline.

Design:
  * Uses the synchronous `google-api-python-client` wrapped in `asyncio.to_thread`.
    The async alternatives are less mature and the polling cadence (~once/minute)
    doesn't need real concurrency.
  * Tracking is label-based: input arrives via the `Fiverr/Orders` label;
    successfully processed emails get `FiverrAgency/Processed`; emails the
    parser refuses get `FiverrAgency/Failed`. The operator can still see the
    raw emails in their normal Gmail flow.
  * Authentication is two-phase: `authorize_interactive` is a one-time CLI
    flow that produces `token.json`; runtime calls use `from_settings` which
    only reads the existing token (and refreshes it transparently).
  * Pure helpers (`_extract_mime_text`, `_strip_html`, `message_to_email`) live
    at module level so they can be unit-tested without instantiating the
    client or hitting the network.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

import structlog

from agency.config import Settings
from agency.intake.email_models import FiverrEmail

if TYPE_CHECKING:  # pragma: no cover
    from googleapiclient.discovery import Resource

logger = structlog.get_logger(__name__)

#: Single scope covers reading messages, adding labels, and creating labels.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]


class GmailClient:
    """Async-facing wrapper around the Gmail REST client.

    Construct via `GmailClient.connect(settings)`. The constructor is private —
    the connect classmethod handles credential loading, label resolution, and
    auto-creates the processed/failed labels if absent.
    """

    def __init__(
        self,
        service: Resource,
        label_pending_id: str,
        label_processed_id: str,
        label_failed_id: str,
        label_processed_search: str,
        label_failed_search: str,
    ) -> None:
        self._service = service
        self._label_pending_id = label_pending_id
        self._label_processed_id = label_processed_id
        self._label_failed_id = label_failed_id
        self._label_processed_search = label_processed_search
        self._label_failed_search = label_failed_search

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    async def connect(cls, settings: Settings) -> GmailClient:
        """Build a client from on-disk token, ensure tracking labels exist.

        Raises if `token.json` is missing or the refresh token is revoked;
        in that case the operator must rerun `agency auth-gmail`.
        """
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build

        def _load_credentials() -> Credentials:
            import os, pathlib
            # Bootstrap credential files from env vars when running in containers.
            for env_var, file_path in [
                ("GMAIL_TOKEN_JSON", settings.gmail_token_file),
                ("GMAIL_CREDENTIALS_JSON", settings.gmail_credentials_file),
            ]:
                val = os.environ.get(env_var)
                if val and not pathlib.Path(file_path).exists():
                    pathlib.Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    pathlib.Path(file_path).write_text(val, encoding="utf-8")

            creds = Credentials.from_authorized_user_file(
                settings.gmail_token_file, GMAIL_SCOPES
            )
            if creds.valid:
                return creds
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                # Persist the refreshed token so subsequent runs don't refresh again immediately.
                with open(settings.gmail_token_file, "w", encoding="utf-8") as fh:
                    fh.write(creds.to_json())
                return creds
            raise RuntimeError(
                f"Gmail token at {settings.gmail_token_file} is invalid. "
                f"Run `agency auth-gmail` to refresh."
            )

        creds = await asyncio.to_thread(_load_credentials)
        service = await asyncio.to_thread(
            build, "gmail", "v1", credentials=creds, cache_discovery=False
        )

        label_ids = await _ensure_labels(
            service,
            [
                settings.gmail_label_pending,
                settings.gmail_label_processed,
                settings.gmail_label_failed,
            ],
        )

        return cls(
            service=service,
            label_pending_id=label_ids[settings.gmail_label_pending],
            label_processed_id=label_ids[settings.gmail_label_processed],
            label_failed_id=label_ids[settings.gmail_label_failed],
            label_processed_search=_label_to_search_token(settings.gmail_label_processed),
            label_failed_search=_label_to_search_token(settings.gmail_label_failed),
        )

    @classmethod
    async def authorize_interactive(cls, settings: Settings) -> None:
        """One-time OAuth bootstrap. Opens a browser, writes `token.json`.

        Requires a local display — run from your workstation, not over headless
        SSH. The resulting `token.json` can then be deployed to the production
        host.
        """
        from google_auth_oauthlib.flow import InstalledAppFlow

        def _run_flow() -> str:
            flow = InstalledAppFlow.from_client_secrets_file(
                settings.gmail_credentials_file, GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)
            return creds.to_json()

        token_json = await asyncio.to_thread(_run_flow)
        with open(settings.gmail_token_file, "w", encoding="utf-8") as fh:
            fh.write(token_json)
        logger.info("gmail.auth_complete", token_file=settings.gmail_token_file)

    # ── Read operations ─────────────────────────────────────────────────

    async def list_pending_message_ids(self, max_results: int = 10) -> list[str]:
        """Return up to `max_results` message ids tagged Pending but not Processed/Failed."""
        query = f"-label:{self._label_processed_search} -label:{self._label_failed_search}"

        def _list() -> dict[str, Any]:
            return (
                self._service.users()
                .messages()
                .list(
                    userId="me",
                    labelIds=[self._label_pending_id],
                    q=query,
                    maxResults=max_results,
                )
                .execute()
            )

        result = await asyncio.to_thread(_list)
        return [m["id"] for m in result.get("messages", [])]

    async def fetch_email(self, message_id: str) -> FiverrEmail:
        """Fetch a full message and convert to a `FiverrEmail`."""

        def _get() -> dict[str, Any]:
            return (
                self._service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )

        message = await asyncio.to_thread(_get)
        return message_to_email(message)

    # ── Write operations ────────────────────────────────────────────────

    async def mark_processed(self, message_id: str) -> None:
        await self._modify_labels(
            message_id,
            add=[self._label_processed_id],
            remove=[self._label_pending_id],
        )

    async def mark_failed(self, message_id: str) -> None:
        await self._modify_labels(
            message_id,
            add=[self._label_failed_id],
            remove=[self._label_pending_id],
        )

    async def _modify_labels(
        self, message_id: str, *, add: list[str], remove: list[str]
    ) -> None:
        def _modify() -> None:
            self._service.users().messages().modify(
                userId="me",
                id=message_id,
                body={"addLabelIds": add, "removeLabelIds": remove},
            ).execute()

        await asyncio.to_thread(_modify)


# ============================================================================
# Pure helpers — unit-testable without any Google client
# ============================================================================


def message_to_email(message: dict[str, Any]) -> FiverrEmail:
    """Convert a Gmail API message dict into a `FiverrEmail`.

    Falls back through preferred body sources:
      1. `text/plain` MIME part
      2. `text/html` MIME part stripped to text
      3. empty string (last resort)
    """
    payload = message.get("payload") or {}
    headers = _headers_lower(payload.get("headers", []))

    body_plain = _extract_mime_text(payload, "text/plain")
    body_html = _extract_mime_text(payload, "text/html")

    if not body_plain and body_html:
        body_plain = _strip_html(body_html)

    internal_date_ms = int(message.get("internalDate", "0"))
    received_at = datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc)

    return FiverrEmail(
        message_id=headers.get("message-id") or message.get("id", ""),
        received_at=received_at,
        subject=headers.get("subject", ""),
        sender=headers.get("from", ""),
        body_plain=body_plain or "",
        body_html=body_html,
        # Gmail attachments are not addressable as URLs without first downloading
        # them and re-uploading to Supabase Storage — out of scope for v1.
        # The parser falls back to URLs in the email body, which Fiverr typically
        # includes as visible links.
        attachment_urls=[],
    )


def _headers_lower(headers: list[dict[str, str]]) -> dict[str, str]:
    """Lowercase header names for case-insensitive lookup."""
    return {h["name"].lower(): h["value"] for h in headers if "name" in h and "value" in h}


def _extract_mime_text(payload: dict[str, Any], mime_type: str) -> str | None:
    """Recursively find a part with the given MIME type and decode its body."""
    if payload.get("mimeType") == mime_type:
        data = (payload.get("body") or {}).get("data")
        if data:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
    for part in payload.get("parts", []) or []:
        result = _extract_mime_text(part, mime_type)
        if result is not None:
            return result
    return None


class _TextExtractor(HTMLParser):
    """Minimal HTML→text. Sufficient for Fiverr notification HTML bodies."""

    def __init__(self) -> None:
        super().__init__()
        self._buf: list[str] = []

    def handle_data(self, data: str) -> None:
        self._buf.append(data)

    def text(self) -> str:
        # Collapse runs of whitespace; preserve word boundaries.
        return " ".join("".join(self._buf).split())


def _strip_html(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return parser.text()


def _label_to_search_token(label_name: str) -> str:
    """Gmail's search syntax replaces `/` with `-` in label tokens.

    Example: `FiverrAgency/Processed` → `FiverrAgency-Processed` for use in
    `q='-label:FiverrAgency-Processed'`.
    """
    return label_name.replace("/", "-")


async def _ensure_labels(service: Resource, names: list[str]) -> dict[str, str]:
    """Look up label ids by name; create labels that do not yet exist.

    Returns a name→id mapping covering exactly `names`.
    """

    def _list_then_create() -> dict[str, str]:
        existing = service.users().labels().list(userId="me").execute()
        existing_by_name = {
            label["name"]: label["id"] for label in existing.get("labels", [])
        }
        result: dict[str, str] = {}
        for name in names:
            if name in existing_by_name:
                result[name] = existing_by_name[name]
                continue
            created = (
                service.users()
                .labels()
                .create(
                    userId="me",
                    body={
                        "name": name,
                        "labelListVisibility": "labelShow",
                        "messageListVisibility": "show",
                    },
                )
                .execute()
            )
            result[name] = created["id"]
            logger.info("gmail.label_created", label=name, id=created["id"])
        return result

    return await asyncio.to_thread(_list_then_create)
