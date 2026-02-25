"""
tests/test_auth_error_copy.py

Unit tests for the _get_auth_error_copy helper defined in app.py.
Tests verify that each gateway error code maps to the correct
(icon_key, headline, body) tuple and that the fallback works correctly.
"""

import sys
import os

# Ensure app module can be imported from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import only the mapping dict and helper — don't trigger full Streamlit init
from app import _AUTH_ERROR_COPY, _AUTH_ERROR_FALLBACK, _get_auth_error_copy


def test_missing_token():
    icon, headline, body = _get_auth_error_copy("missing_token")
    assert icon == "shield-alert"
    assert "연결이 필요합니다" in headline
    assert len(body) > 0


def test_token_expired():
    icon, headline, body = _get_auth_error_copy("token_expired")
    assert icon == "clock"
    assert "시간이" in headline
    assert len(body) > 0


def test_forbidden_audience():
    icon, headline, body = _get_auth_error_copy("forbidden_audience")
    assert icon == "lock"
    assert "접근 권한" in headline
    assert "열쇠" in body


def test_forbidden_issuer():
    icon, headline, body = _get_auth_error_copy("forbidden_issuer")
    assert icon == "lock"
    assert "접근 권한" in headline


def test_invalid_token():
    icon, headline, body = _get_auth_error_copy("invalid_token")
    assert icon == "zap-off"
    assert "기억" in headline
    assert len(body) > 0


def test_invalid_authorization():
    icon, headline, body = _get_auth_error_copy("invalid_authorization")
    assert icon == "zap-off"
    assert "기억" in headline


def test_fallback_unknown_code():
    """Any unknown code should return the safe fallback."""
    icon, headline, body = _get_auth_error_copy("some_unknown_code")
    assert icon == _AUTH_ERROR_FALLBACK[0]
    assert headline == _AUTH_ERROR_FALLBACK[1]
    assert body == _AUTH_ERROR_FALLBACK[2]


def test_empty_code_uses_fallback():
    icon, headline, body = _get_auth_error_copy("")
    assert icon == _AUTH_ERROR_FALLBACK[0]


def test_all_mapped_codes_have_valid_structure():
    """All mapping entries must be 3-tuples with non-empty strings."""
    for code, entry in _AUTH_ERROR_COPY.items():
        assert len(entry) == 3, f"Code '{code}' mapping must be a 3-tuple"
        icon, headline, body = entry
        assert isinstance(icon, str) and len(icon) > 0, f"Code '{code}' missing icon"
        assert isinstance(headline, str) and len(headline) > 0, f"Code '{code}' missing headline"
        assert isinstance(body, str) and len(body) > 0, f"Code '{code}' missing body"
