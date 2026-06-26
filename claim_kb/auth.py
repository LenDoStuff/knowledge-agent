"""Microsoft Entra browser authentication helpers."""

from __future__ import annotations

from typing import Any

from claim_kb.config import ClaimKbSettings


def create_browser_credential(settings: ClaimKbSettings) -> Any:
    """Create an InteractiveBrowserCredential, honoring AZURE_TENANT_ID if set."""

    from azure.identity import InteractiveBrowserCredential

    kwargs: dict[str, str] = {}
    if settings.tenant_id:
        kwargs["tenant_id"] = settings.tenant_id
    return InteractiveBrowserCredential(**kwargs)
