"""DLP (Data Loss Prevention) guard — scans outgoing messages for sensitive
data patterns and either redacts them or blocks the request before it
leaves the local machine.

Runs as pre-middleware so it inspects every message *before* it hits
any adapter.  When the target adapter is local (localhost/127.0.0.1),
the guard is skipped by default since data stays on-machine.

Detectable patterns:
  - API keys / tokens (AWS, GitHub, Slack, generic Bearer tokens)
  - Private keys (RSA, EC, PGP, SSH)
  - Credit card numbers (Visa, MC, Amex, Discover)
  - US Social Security Numbers
  - Email addresses (optional — off by default)
  - IPv4 addresses with private ranges (optional — off by default)
  - Custom regex patterns via config

Actions:
  - "block"  — reject the request with an error (default)
  - "redact" — replace matches with [REDACTED] and continue
"""

from __future__ import annotations

import logging
import re
from typing import Any

from src.core.models import Choice, Message, PipelineResponse, Role
from src.core.state import PipelineState
from src.middleware.base import PreMiddleware
from src.utils.errors import RequestValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Built-in patterns — each is (name, compiled regex)
# ---------------------------------------------------------------------------

_BUILTIN_PATTERNS: list[tuple[str, re.Pattern]] = [
    # AWS access key ID
    (
        "AWS Access Key",
        re.compile(r"(?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])"),
    ),
    # AWS secret key (40-char base64 after common prefixes)
    (
        "AWS Secret Key",
        re.compile(
            r"(?:aws_secret_access_key|secret_key|SECRET_KEY)"
            r"[\s:=\"']+([A-Za-z0-9/+=]{40})"
        ),
    ),
    # GitHub personal access token (classic and fine-grained)
    (
        "GitHub Token",
        re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,255}"),
    ),
    # Slack tokens
    (
        "Slack Token",
        re.compile(r"xox[bpors]-[0-9a-zA-Z\-]{10,250}"),
    ),
    # Generic high-entropy API key patterns (key=..., token=..., secret=...)
    (
        "Generic API Key",
        re.compile(
            r"(?:api[_-]?key|api[_-]?secret|access[_-]?token|secret[_-]?key|auth[_-]?token)"
            r"[\s]*[=:]\s*['\"]?([A-Za-z0-9\-_.]{20,})['\"]?",
            re.IGNORECASE,
        ),
    ),
    # Bearer tokens in text
    (
        "Bearer Token",
        re.compile(r"Bearer\s+[A-Za-z0-9\-_=]{20,}"),
    ),
    # Private keys (PEM format)
    (
        "Private Key",
        re.compile(
            r"-----BEGIN\s+(?:RSA|EC|DSA|OPENSSH|PGP)?\s*PRIVATE\s+KEY-----"
        ),
    ),
    # Credit card numbers (Visa, MC, Amex, Discover — with optional separators)
    (
        "Credit Card Number",
        re.compile(
            r"(?<!\d)"
            r"(?:4[0-9]{3}|5[1-5][0-9]{2}|3[47][0-9]{2}|6(?:011|5[0-9]{2}))"
            r"[\s\-]?[0-9]{4}[\s\-]?[0-9]{4}[\s\-]?[0-9]{1,4}"
            r"(?!\d)"
        ),
    ),
    # US Social Security Number
    (
        "SSN",
        re.compile(
            r"(?<!\d)(?!000|666|9\d{2})\d{3}"
            r"[\s\-](?!00)\d{2}[\s\-](?!0000)\d{4}(?!\d)"
        ),
    ),
    # Password assignments in text
    (
        "Password",
        re.compile(
            r"(?:password|passwd|pwd)\s*[=:]\s*['\"]?(\S{6,})['\"]?",
            re.IGNORECASE,
        ),
    ),
]

# Optional patterns — not in the default set
_OPTIONAL_PATTERNS: dict[str, tuple[str, re.Pattern]] = {
    "email": (
        "Email Address",
        re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    ),
    "ipv4": (
        "IPv4 Address",
        re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    ),
}

# Hosts considered "local" — requests to these skip the guard
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}


def _is_local_adapter(adapters: dict, adapter_name: str) -> bool:
    """Check if the target adapter points to a local endpoint."""
    adapter = adapters.get(adapter_name)
    if adapter is None:
        return False
    base_url = getattr(adapter, "base_url", "")
    if not base_url:
        return False
    # Extract host from URL
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        host = parsed.hostname or ""
        return host in _LOCAL_HOSTS
    except Exception:
        return False


class DLPGuardMiddleware(PreMiddleware):
    """Pre-middleware that blocks or redacts sensitive data in messages."""

    name = "dlp_guard"

    def __init__(
        self,
        action: str = "block",
        skip_local: bool = True,
        extra_patterns: list[dict[str, str]] | None = None,
        enable_optional: list[str] | None = None,
    ) -> None:
        """
        Args:
            action: "block" to reject, "redact" to replace matches.
            skip_local: If True, skip scanning when adapter is localhost.
            extra_patterns: List of {"name": ..., "pattern": ...} custom regexes.
            enable_optional: List of optional pattern keys to enable (e.g. ["email"]).
        """
        if action not in ("block", "redact"):
            raise ValueError(f"DLP action must be 'block' or 'redact', got '{action}'")
        self.action = action
        self.skip_local = skip_local

        # Build the active pattern list
        self.patterns: list[tuple[str, re.Pattern]] = list(_BUILTIN_PATTERNS)

        for key in (enable_optional or []):
            if key in _OPTIONAL_PATTERNS:
                self.patterns.append(_OPTIONAL_PATTERNS[key])

        for p in (extra_patterns or []):
            self.patterns.append((p["name"], re.compile(p["pattern"])))

    async def process(self, state: PipelineState) -> PipelineState:
        if state.response is not None:
            return state

        # Skip if targeting a local adapter
        if self.skip_local:
            adapter_name = state.request.target_adapter
            pipeline = state.extras.get("_pipeline")
            # Also check via the adapters dict passed through extras
            # The middleware doesn't have direct pipeline access, so we
            # check the base_url on the adapter if we can find it
            if adapter_name:
                adapters = state.extras.get("_adapters", {})
                if _is_local_adapter(adapters, adapter_name):
                    logger.debug(
                        "DLP guard: skipping scan — adapter '%s' is local",
                        adapter_name,
                    )
                    return state

        # Scan all user messages
        findings: list[tuple[str, str]] = []  # (pattern_name, matched_text)

        for msg in state.request.messages:
            if msg.role not in (Role.USER, Role.SYSTEM):
                continue
            text = msg.content
            if not text:
                continue

            for pattern_name, pattern_re in self.patterns:
                matches = pattern_re.findall(text)
                if matches:
                    for m in matches:
                        match_str = m if isinstance(m, str) else m[0] if m else ""
                        findings.append((pattern_name, match_str[:30]))

        if not findings:
            return state

        # --- Sensitive data detected ---
        pattern_names = sorted({f[0] for f in findings})
        logger.warning(
            "DLP guard: detected %d sensitive pattern(s): %s",
            len(findings),
            ", ".join(pattern_names),
        )

        if self.action == "block":
            raise RequestValidationError(
                f"Request blocked — sensitive data detected: "
                f"{', '.join(pattern_names)}. "
                f"Remove sensitive content before sending to an external AI.",
            )

        # action == "redact"
        for msg in state.request.messages:
            if msg.role not in (Role.USER, Role.SYSTEM):
                continue
            if not msg.content:
                continue
            for pattern_name, pattern_re in self.patterns:
                msg.content = pattern_re.sub(
                    f"[REDACTED:{pattern_name}]", msg.content
                )

        state.extras["dlp_redacted"] = True
        state.extras["dlp_patterns_found"] = pattern_names
        logger.info(
            "DLP guard: redacted %d pattern(s) in request %s",
            len(findings),
            state.request.id,
        )

        return state
