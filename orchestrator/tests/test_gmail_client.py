"""Unit tests for the pure Gmail helpers.

The `GmailClient` class itself is mostly an async wrapper around the official
SDK and is exercised end-to-end via `test_runner.py` with a mocked client.
This file targets the pure functions — MIME extraction, HTML stripping,
message → FiverrEmail conversion — which carry all the real parsing risk.
"""

from __future__ import annotations

import base64

from agency.intake.email_models import FiverrEmail
from agency.intake.gmail_client import (
    _extract_mime_text,
    _label_to_search_token,
    _strip_html,
    message_to_email,
)


def _b64(text: str) -> str:
    """URL-safe base64 encode (Gmail's format)."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


# ── _extract_mime_text ──────────────────────────────────────────────────────


def test_extract_mime_text_flat_part():
    payload = {"mimeType": "text/plain", "body": {"data": _b64("hello")}}
    assert _extract_mime_text(payload, "text/plain") == "hello"


def test_extract_mime_text_nested_part():
    payload = {
        "mimeType": "multipart/alternative",
        "parts": [
            {"mimeType": "text/html", "body": {"data": _b64("<p>hi</p>")}},
            {"mimeType": "text/plain", "body": {"data": _b64("hi plain")}},
        ],
    }
    assert _extract_mime_text(payload, "text/plain") == "hi plain"
    assert _extract_mime_text(payload, "text/html") == "<p>hi</p>"


def test_extract_mime_text_deeply_nested():
    """Real Gmail messages can nest multipart/mixed → multipart/alternative."""
    payload = {
        "mimeType": "multipart/mixed",
        "parts": [
            {
                "mimeType": "multipart/alternative",
                "parts": [
                    {"mimeType": "text/plain", "body": {"data": _b64("buried plain")}},
                ],
            }
        ],
    }
    assert _extract_mime_text(payload, "text/plain") == "buried plain"


def test_extract_mime_text_returns_none_when_absent():
    payload = {"mimeType": "text/html", "body": {"data": _b64("<p>x</p>")}}
    assert _extract_mime_text(payload, "text/plain") is None


def test_extract_mime_text_handles_missing_body_data():
    payload = {"mimeType": "text/plain", "body": {}}
    assert _extract_mime_text(payload, "text/plain") is None


# ── _strip_html ─────────────────────────────────────────────────────────────


def test_strip_html_removes_tags():
    html = "<html><body><p>Hello <b>world</b></p></body></html>"
    assert _strip_html(html) == "Hello world"


def test_strip_html_collapses_whitespace():
    html = "<p>multiple   \n  spaces</p>"
    assert _strip_html(html) == "multiple spaces"


def test_strip_html_handles_empty_input():
    assert _strip_html("") == ""


# ── _label_to_search_token ──────────────────────────────────────────────────


def test_label_to_search_token_replaces_slashes_with_hyphens():
    assert _label_to_search_token("FiverrAgency/Processed") == "FiverrAgency-Processed"


def test_label_to_search_token_passes_through_flat_names():
    assert _label_to_search_token("INBOX") == "INBOX"


# ── message_to_email ────────────────────────────────────────────────────────


def _build_message(
    *,
    message_id: str = "<abc@gmail>",
    subject: str = "New order",
    sender: str = "noreply@fiverr.com",
    plain_body: str | None = "Plain body",
    html_body: str | None = None,
    internal_date_ms: int = 1747315200000,  # 2025-05-15T12:00:00Z
) -> dict:
    parts = []
    if plain_body is not None:
        parts.append({"mimeType": "text/plain", "body": {"data": _b64(plain_body)}})
    if html_body is not None:
        parts.append({"mimeType": "text/html", "body": {"data": _b64(html_body)}})

    return {
        "id": "gmail-internal-id",
        "internalDate": str(internal_date_ms),
        "payload": {
            "headers": [
                {"name": "Message-ID", "value": message_id},
                {"name": "Subject", "value": subject},
                {"name": "From", "value": sender},
            ],
            "mimeType": "multipart/alternative",
            "parts": parts,
        },
    }


def test_message_to_email_prefers_plain_body():
    message = _build_message(plain_body="plain text", html_body="<p>html</p>")
    email = message_to_email(message)
    assert isinstance(email, FiverrEmail)
    assert email.body_plain == "plain text"
    assert email.body_html == "<p>html</p>"


def test_message_to_email_falls_back_to_stripped_html():
    """If only HTML is present, body_plain comes from stripping the HTML."""
    message = _build_message(plain_body=None, html_body="<p>Hello <b>world</b></p>")
    email = message_to_email(message)
    assert email.body_plain == "Hello world"
    assert email.body_html == "<p>Hello <b>world</b></p>"


def test_message_to_email_falls_back_to_internal_id_when_no_message_id_header():
    message = _build_message()
    # Drop the Message-ID header
    message["payload"]["headers"] = [
        h for h in message["payload"]["headers"] if h["name"].lower() != "message-id"
    ]
    email = message_to_email(message)
    assert email.message_id == message["id"]


def test_message_to_email_case_insensitive_headers():
    message = _build_message()
    # Lowercase the header name — Gmail sometimes does this
    message["payload"]["headers"] = [
        {"name": "message-id", "value": "<lowercase@gmail>"},
        {"name": "Subject", "value": "subject"},
        {"name": "From", "value": "from@example.com"},
    ]
    email = message_to_email(message)
    assert email.message_id == "<lowercase@gmail>"


def test_message_to_email_received_at_in_utc():
    message = _build_message(internal_date_ms=1747315200000)
    email = message_to_email(message)
    assert email.received_at.tzinfo is not None
    assert email.received_at.isoformat() == "2025-05-15T12:00:00+00:00"


def test_message_to_email_handles_missing_payload():
    """Edge case: a malformed Gmail response missing the payload entirely."""
    message = {"id": "x", "internalDate": "0"}
    email = message_to_email(message)
    assert email.message_id == "x"
    assert email.body_plain == ""
    assert email.subject == ""
