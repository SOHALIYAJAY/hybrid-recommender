"""Tests for SSRF protection on dataset URL validation."""

from unittest.mock import patch

import pytest

from backend.url_validation import (
    DISALLOWED_URL_MESSAGE,
    UrlValidationError,
    validate_public_url,
)


def _public_addrinfo(host: str = "example.com", ip: str = "93.184.216.34", port: int = 443):
    import socket

    return [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)),
    ]


@patch("backend.url_validation.socket.getaddrinfo", return_value=_public_addrinfo())
def test_validate_public_url_allows_https_example_com(mock_getaddrinfo):
    validate_public_url("https://example.com")
    mock_getaddrinfo.assert_called_once()


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost",
        "http://127.0.0.1",
        "http://10.0.0.1",
        "http://192.168.1.1",
        "http://169.254.169.254",
        "http://[::1]",
    ],
)
def test_validate_public_url_blocks_internal_targets(url):
    with pytest.raises(UrlValidationError, match=DISALLOWED_URL_MESSAGE):
        validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/data.csv",
        "gopher://example.com/",
        "data:text/plain,hello",
    ],
)
def test_validate_public_url_blocks_non_http_schemes(url):
    with pytest.raises(UrlValidationError, match=DISALLOWED_URL_MESSAGE):
        validate_public_url(url)


@patch("backend.url_validation.socket.getaddrinfo", return_value=_public_addrinfo(ip="10.0.0.1"))
def test_validate_public_url_blocks_resolved_private_ips(mock_getaddrinfo):
    with pytest.raises(UrlValidationError, match=DISALLOWED_URL_MESSAGE):
        validate_public_url("https://example.com")


# ---------------------------------------------------------------------------
# Tests added for Issue #393 review gaps
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:127.0.0.1]",
        "http://[::ffff:169.254.169.254]",
        "http://[::ffff:10.0.0.1]",
        "http://[::ffff:192.168.1.1]",
    ],
)
def test_validate_public_url_blocks_ipv4_mapped_ipv6(url):
    """IPv4-mapped IPv6 addresses must be blocked on all Python versions (incl. 3.10)."""
    with pytest.raises(UrlValidationError, match=DISALLOWED_URL_MESSAGE):
        validate_public_url(url)


def test_validate_public_url_blocks_cgnat():
    """Carrier-Grade NAT range 100.64.0.0/10 must be blocked."""
    with pytest.raises(UrlValidationError, match=DISALLOWED_URL_MESSAGE):
        validate_public_url("http://100.64.0.1")


def test_validate_public_url_blocks_ipv6_aws_metadata():
    """IPv6 AWS EC2 metadata endpoint fd00:ec2::254 must be blocked."""
    with pytest.raises(UrlValidationError, match=DISALLOWED_URL_MESSAGE):
        validate_public_url("http://[fd00:ec2::254]")


def test_redirect_validator_blocks_private_redirect():
    """_RedirectValidator must reject redirects that point to internal destinations."""
    from backend.dataset_url_fetcher import _RedirectValidator

    handler = _RedirectValidator()
    with pytest.raises(UrlValidationError, match=DISALLOWED_URL_MESSAGE):
        handler.redirect_request(None, None, 302, "Found", {}, "http://192.168.1.1/secret")
