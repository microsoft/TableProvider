import hmac
import os
import re

from flask import abort, request


def protect_requests(app):
    token = app.config.get("TABLEPROVIDER_API_TOKEN") or os.environ.get("TABLEPROVIDER_API_TOKEN")
    if (not isinstance(token, str) or not 32 <= len(token) <= 512
            or not token.isascii() or any(character.isspace() for character in token)):
        raise RuntimeError("Set TABLEPROVIDER_API_TOKEN to a private random ASCII token of at least 32 characters")
    hosts = app.config.get("TABLEPROVIDER_ALLOWED_HOSTS")
    if hosts is None:
        hosts = os.environ.get("TABLEPROVIDER_ALLOWED_HOSTS", "127.0.0.1:8080,localhost:8080,[::1]:8080")
    if isinstance(hosts, str):
        hosts = hosts.split(",")
    if not isinstance(hosts, (list, tuple)) or not hosts:
        raise RuntimeError("TABLEPROVIDER_ALLOWED_HOSTS must contain exact host names and ports")
    allowed = set()
    for host in hosts:
        if not isinstance(host, str):
            raise RuntimeError("Invalid allowed host")
        host = host.strip().lower()
        if not re.fullmatch(r"(?:\[::1\]|[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?)(?::[0-9]{1,5})?", host):
            raise RuntimeError("Allowed hosts must be exact names, not URLs or wildcards")
        allowed.add(host)
    app.config["MAX_CONTENT_LENGTH"] = app.config.get("MAX_CONTENT_LENGTH") or 1024 * 1024

    @app.before_request
    def require_trusted_request():
        if request.host.lower() not in allowed:
            abort(400, description="Untrusted Host header")
        site = request.headers.get("Sec-Fetch-Site", "")
        if site not in ("", "same-origin", "none"):
            abort(403, description="Cross-origin browser requests are not allowed")
        origin = request.headers.get("Origin")
        if origin is not None and origin.lower() != f"{request.scheme}://{request.host}".lower():
            abort(403, description="Untrusted Origin header")
        authorization = request.headers.get("Authorization", "")
        supplied = authorization[7:] if authorization.startswith("Bearer ") else ""
        if not hmac.compare_digest(supplied.encode("utf-8"), token.encode("ascii")):
            abort(401, description="Bearer authentication required")
        if request.content_length is not None and request.content_length > app.config["MAX_CONTENT_LENGTH"]:
            abort(413, description="Request body exceeds the size limit")