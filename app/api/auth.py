"""API-key authentication and role-based access control.

Scope decision: API-key auth with a role baked into the key record from
day one, not full JWT -- appropriate for a single-operator showcase app.
Storing role/allowed_scope on the key now lets it reuse cleanly as JWT
claims later without a second schema migration.
"""

import hashlib
import logging
import secrets
from uuid import UUID

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import AUTH_API_KEY_SALT, AUTH_BOOTSTRAP_ADMIN_KEY

from .schemas import ApiKeyRecord, ApiKeyRole
from .store import RunStore

logger = logging.getLogger(__name__)

_bearer_scheme = HTTPBearer(auto_error=False)


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256((AUTH_API_KEY_SALT + raw_key).encode("utf-8")).hexdigest()


def generate_api_key() -> str:
    return secrets.token_urlsafe(32)


async def require_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> ApiKeyRecord:
    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    store: RunStore = request.app.state.run_service.store
    key_record = await store.get_api_key_by_hash(hash_api_key(credentials.credentials))

    if key_record is None or key_record.revoked_at is not None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    return key_record


async def require_admin(
    key: ApiKeyRecord = Depends(require_api_key),
) -> ApiKeyRecord:
    if key.role != ApiKeyRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin role required")
    return key


def is_authorized_owner(key: ApiKeyRecord, owner_key_id: UUID | None) -> bool:
    """Fail-closed: a resource with no owner (legacy/pre-auth) is visible
    only to admins, not to every ordinary key."""

    if key.role == ApiKeyRole.ADMIN:
        return True
    return owner_key_id is not None and owner_key_id == key.id


async def ensure_bootstrap_admin_key(store: RunStore) -> None:
    """Idempotently seed an admin key from AUTH_BOOTSTRAP_ADMIN_KEY so the
    system can be administered without a chicken-and-egg problem. Once
    seeded, the bootstrap key behaves like any other key (revocable)."""

    if not AUTH_BOOTSTRAP_ADMIN_KEY:
        return

    hashed = hash_api_key(AUTH_BOOTSTRAP_ADMIN_KEY)
    if await store.get_api_key_by_hash(hashed) is not None:
        return

    await store.create_api_key(
        ApiKeyRecord(hashed_key=hashed, label="bootstrap", role=ApiKeyRole.ADMIN)
    )
    logger.info("Seeded bootstrap admin API key")
