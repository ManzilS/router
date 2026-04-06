"""Security tests for the early-exit middleware's math evaluator."""

import pytest

from src.middleware.early_exit import _safe_math_eval


def test_basic_addition():
    assert _safe_math_eval("2 + 3") == 5


def test_order_of_operations():
    assert _safe_math_eval("2 + 3 * 4") == 14


def test_parentheses():
    assert _safe_math_eval("(2 + 3) * 4") == 20


def test_float_division():
    result = _safe_math_eval("10 / 3")
    assert abs(result - 3.3333333) < 0.001


def test_negative_numbers():
    assert _safe_math_eval("-5 + 3") == -2


def test_division_by_zero_returns_none():
    assert _safe_math_eval("1 / 0") is None


def test_syntax_error_returns_none():
    assert _safe_math_eval("2 +") is None


def test_rejects_function_calls():
    """Should reject anything that isn't pure arithmetic."""
    assert _safe_math_eval("__import__('os').system('rm -rf /')") is None


def test_rejects_attribute_access():
    assert _safe_math_eval("().__class__") is None


def test_rejects_string_literals():
    assert _safe_math_eval("'hello'") is None


def test_complex_expression():
    assert _safe_math_eval("(10 + 5) * 2 - 3") == 27


def test_nested_parens():
    assert _safe_math_eval("((2 + 3) * (4 - 1))") == 15
