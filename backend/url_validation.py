"""SSRF-safe validation for user-supplied dataset URLs."""

from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

DISALLOWED_URL_MESSAGE = "Invalid or disallowed URL"


class UrlValidationError(ValueError):
    """Raised when a URL is missing, malformed, or targets a disallowed destination."""


def validate_public_url(url: str) -> None:
    """
    Parse *url*, allow only http/https, resolve the hostname, and reject
    private/internal/reserved destinations.
    """
    if not url or not isinstance(url, str):
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    hostname = parsed.hostname
    if not hostname:
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    if parsed.username or parsed.password:
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    host = hostname.strip().lower().rstrip(".")
    if host == "localhost":
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    port = parsed.port
    if port is not None and not (1 <= port <= 65535):
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    if host.startswith("[") and host.endswith("]"):
        literal = host[1:-1]
    else:
        literal = host

    try:
        _reject_ip(ipaddress.ip_address(literal))
        return
    except ValueError:
        pass

    try:
        addrinfos = socket.getaddrinfo(
            host,
            port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except OSError:
        raise UrlValidationError(DISALLOWED_URL_MESSAGE) from None

    if not addrinfos:
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    for info in addrinfos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        _reject_ip(ipaddress.ip_address(sockaddr[0]))


def _reject_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> None:
    # Unwrap IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1) and validate
    # the embedded IPv4 address. Python 3.10 does not set is_loopback/is_private
    # on these addresses, so the unwrap must happen before the flag checks.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        _reject_ip(ip.ipv4_mapped)

    if (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    if isinstance(ip, ipaddress.IPv4Address):
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            raise UrlValidationError(DISALLOWED_URL_MESSAGE)
        if ip == ipaddress.ip_address("169.254.169.254"):
            raise UrlValidationError(DISALLOWED_URL_MESSAGE)

    if ip == ipaddress.ip_address("fd00:ec2::254"):
        raise UrlValidationError(DISALLOWED_URL_MESSAGE)
