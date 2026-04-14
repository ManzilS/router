"""Tests for the DLP guard middleware — sensitive data detection."""

import pytest

from src.core.models import Message, PipelineRequest, Role
from src.core.state import PipelineState
from src.middleware.dlp_guard import DLPGuardMiddleware, _is_local_adapter
from src.utils.errors import RequestValidationError


def _state(content: str, adapter: str = "openai") -> PipelineState:
    return PipelineState(
        request=PipelineRequest(
            messages=[Message(role=Role.USER, content=content)],
            target_adapter=adapter,
        )
    )


# ---------------------------------------------------------------------------
# Block mode (default)
# ---------------------------------------------------------------------------

class TestBlockMode:
    @pytest.fixture
    def guard(self):
        return DLPGuardMiddleware(action="block", skip_local=False)

    @pytest.mark.asyncio
    async def test_blocks_aws_access_key(self, guard):
        state = _state("My key is AKIAIOSFODNN7EXAMPLE")
        with pytest.raises(RequestValidationError, match="AWS Access Key"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_aws_secret_key(self, guard):
        state = _state("aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
        with pytest.raises(RequestValidationError, match="AWS Secret Key"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_github_token(self, guard):
        state = _state("token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl")
        with pytest.raises(RequestValidationError, match="GitHub Token"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_slack_token(self, guard):
        state = _state("SLACK_TOKEN=xoxb-1234567890-abcdefghij")
        with pytest.raises(RequestValidationError, match="Slack Token"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_private_key(self, guard):
        state = _state("-----BEGIN RSA PRIVATE KEY-----\nMIIEow...")
        with pytest.raises(RequestValidationError, match="Private Key"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_openssh_private_key(self, guard):
        state = _state("-----BEGIN OPENSSH PRIVATE KEY-----\nb3Blbn...")
        with pytest.raises(RequestValidationError, match="Private Key"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_credit_card_visa(self, guard):
        state = _state("My card is 4111 1111 1111 1111")
        with pytest.raises(RequestValidationError, match="Credit Card"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_credit_card_mastercard(self, guard):
        state = _state("Card: 5500-0000-0000-0004")
        with pytest.raises(RequestValidationError, match="Credit Card"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_credit_card_amex(self, guard):
        state = _state("Amex: 378282246310005")
        with pytest.raises(RequestValidationError, match="Credit Card"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_ssn(self, guard):
        state = _state("SSN: 123-45-6789")
        with pytest.raises(RequestValidationError, match="SSN"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_ssn_with_spaces(self, guard):
        state = _state("SSN is 123 45 6789")
        with pytest.raises(RequestValidationError, match="SSN"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_bearer_token(self, guard):
        state = _state("Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6")
        with pytest.raises(RequestValidationError, match="Bearer Token"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_generic_api_key(self, guard):
        state = _state("api_key = my_secret_key_abcdefghijklmnop")
        with pytest.raises(RequestValidationError, match="API Key"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_blocks_password_assignment(self, guard):
        state = _state('password = "SuperSecret123!"')
        with pytest.raises(RequestValidationError, match="Password"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_allows_clean_message(self, guard):
        state = _state("How do I write a Python function?")
        result = await guard.process(state)
        assert result.response is None
        assert result.early_exit is False

    @pytest.mark.asyncio
    async def test_allows_short_numbers(self, guard):
        """Short numbers that don't match patterns should pass."""
        state = _state("The answer is 42")
        result = await guard.process(state)
        assert result.response is None

    @pytest.mark.asyncio
    async def test_scans_system_messages(self, guard):
        state = PipelineState(
            request=PipelineRequest(
                messages=[
                    Message(role=Role.SYSTEM, content="api_key = my_secret_key_abcdefghijklmnop"),
                    Message(role=Role.USER, content="Hello"),
                ],
                target_adapter="openai",
            )
        )
        with pytest.raises(RequestValidationError, match="API Key"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_skips_assistant_messages(self, guard):
        """DLP should not scan assistant (AI-generated) messages."""
        state = PipelineState(
            request=PipelineRequest(
                messages=[
                    Message(role=Role.ASSISTANT, content="ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijkl"),
                    Message(role=Role.USER, content="Thanks"),
                ],
                target_adapter="openai",
            )
        )
        result = await guard.process(state)
        assert result.response is None


# ---------------------------------------------------------------------------
# Redact mode
# ---------------------------------------------------------------------------

class TestRedactMode:
    @pytest.fixture
    def guard(self):
        return DLPGuardMiddleware(action="redact", skip_local=False)

    @pytest.mark.asyncio
    async def test_redacts_aws_key(self, guard):
        state = _state("Key is AKIAIOSFODNN7EXAMPLE, use it")
        result = await guard.process(state)
        content = result.request.messages[0].content
        assert "AKIAIOSFODNN7EXAMPLE" not in content
        assert "[REDACTED:AWS Access Key]" in content
        assert result.extras.get("dlp_redacted") is True

    @pytest.mark.asyncio
    async def test_redacts_multiple_patterns(self, guard):
        state = _state(
            "Key: AKIAIOSFODNN7EXAMPLE, SSN: 123-45-6789"
        )
        result = await guard.process(state)
        content = result.request.messages[0].content
        assert "AKIAIOSFODNN7EXAMPLE" not in content
        assert "123-45-6789" not in content
        assert "AWS Access Key" in result.extras["dlp_patterns_found"]
        assert "SSN" in result.extras["dlp_patterns_found"]

    @pytest.mark.asyncio
    async def test_redact_preserves_safe_text(self, guard):
        state = _state("Hello AKIAIOSFODNN7EXAMPLE world")
        result = await guard.process(state)
        content = result.request.messages[0].content
        assert content.startswith("Hello ")
        assert content.endswith(" world")

    @pytest.mark.asyncio
    async def test_clean_message_unchanged(self, guard):
        state = _state("Just a normal question about Python")
        result = await guard.process(state)
        assert result.request.messages[0].content == "Just a normal question about Python"
        assert result.extras.get("dlp_redacted") is None


# ---------------------------------------------------------------------------
# Skip local
# ---------------------------------------------------------------------------

class TestSkipLocal:
    @pytest.mark.asyncio
    async def test_skips_local_adapter(self):
        guard = DLPGuardMiddleware(action="block", skip_local=True)

        class FakeAdapter:
            base_url = "http://localhost:1234/v1"

        state = _state("AKIAIOSFODNN7EXAMPLE", adapter="lmstudio")
        state.extras["_adapters"] = {"lmstudio": FakeAdapter()}
        result = await guard.process(state)
        # Should NOT block — adapter is local
        assert result.response is None

    @pytest.mark.asyncio
    async def test_does_not_skip_remote_adapter(self):
        guard = DLPGuardMiddleware(action="block", skip_local=True)

        class FakeAdapter:
            base_url = "https://api.openai.com/v1"

        state = _state("AKIAIOSFODNN7EXAMPLE", adapter="openai")
        state.extras["_adapters"] = {"openai": FakeAdapter()}
        with pytest.raises(RequestValidationError, match="AWS Access Key"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_skip_local_false_scans_everything(self):
        guard = DLPGuardMiddleware(action="block", skip_local=False)
        state = _state("AKIAIOSFODNN7EXAMPLE", adapter="lmstudio")
        with pytest.raises(RequestValidationError, match="AWS Access Key"):
            await guard.process(state)


# ---------------------------------------------------------------------------
# Optional patterns
# ---------------------------------------------------------------------------

class TestOptionalPatterns:
    @pytest.mark.asyncio
    async def test_email_not_detected_by_default(self):
        guard = DLPGuardMiddleware(action="block", skip_local=False)
        state = _state("Contact me at user@example.com")
        result = await guard.process(state)
        assert result.response is None

    @pytest.mark.asyncio
    async def test_email_detected_when_enabled(self):
        guard = DLPGuardMiddleware(
            action="block", skip_local=False, enable_optional=["email"]
        )
        state = _state("Contact me at user@example.com")
        with pytest.raises(RequestValidationError, match="Email"):
            await guard.process(state)


# ---------------------------------------------------------------------------
# Custom patterns
# ---------------------------------------------------------------------------

class TestCustomPatterns:
    @pytest.mark.asyncio
    async def test_custom_pattern_detected(self):
        guard = DLPGuardMiddleware(
            action="block",
            skip_local=False,
            extra_patterns=[{"name": "Internal ID", "pattern": r"PROJ-\d{4,}"}],
        )
        state = _state("See ticket PROJ-12345 for details")
        with pytest.raises(RequestValidationError, match="Internal ID"):
            await guard.process(state)

    @pytest.mark.asyncio
    async def test_custom_pattern_no_match(self):
        guard = DLPGuardMiddleware(
            action="block",
            skip_local=False,
            extra_patterns=[{"name": "Internal ID", "pattern": r"PROJ-\d{4,}"}],
        )
        state = _state("Just a normal message")
        result = await guard.process(state)
        assert result.response is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_empty_message(self):
        guard = DLPGuardMiddleware(action="block", skip_local=False)
        state = _state("")
        result = await guard.process(state)
        assert result.response is None

    @pytest.mark.asyncio
    async def test_skips_when_response_already_set(self):
        guard = DLPGuardMiddleware(action="block", skip_local=False)
        from src.core.models import Choice, PipelineResponse

        state = _state("AKIAIOSFODNN7EXAMPLE")
        state.response = PipelineResponse(
            choices=[Choice(message=Message(role=Role.ASSISTANT, content="done"))]
        )
        result = await guard.process(state)
        # Should skip processing — response already exists
        assert result.response is not None

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="block.*redact"):
            DLPGuardMiddleware(action="warn")

    @pytest.mark.asyncio
    async def test_ssn_rejects_invalid_formats(self):
        """000, 666, and 9xx prefixes are invalid SSNs."""
        guard = DLPGuardMiddleware(action="block", skip_local=False)
        for invalid in ["000-12-3456", "666-12-3456", "900-12-3456"]:
            state = _state(f"SSN: {invalid}")
            result = await guard.process(state)
            assert result.response is None  # should NOT trigger


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

class TestIsLocalAdapter:
    def test_localhost(self):
        class A:
            base_url = "http://localhost:1234/v1"
        assert _is_local_adapter({"a": A()}, "a") is True

    def test_127_0_0_1(self):
        class A:
            base_url = "http://127.0.0.1:8080"
        assert _is_local_adapter({"a": A()}, "a") is True

    def test_remote(self):
        class A:
            base_url = "https://api.openai.com/v1"
        assert _is_local_adapter({"a": A()}, "a") is False

    def test_missing_adapter(self):
        assert _is_local_adapter({}, "missing") is False

    def test_no_base_url(self):
        class A:
            pass
        assert _is_local_adapter({"a": A()}, "a") is False
