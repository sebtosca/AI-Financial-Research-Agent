from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from app.api import auth as auth_module
from app.api.auth import (
    ensure_bootstrap_admin_key,
    generate_api_key,
    hash_api_key,
    is_authorized_owner,
    require_admin,
    require_api_key,
)
from app.api.schemas import ApiKeyRecord, ApiKeyRole
from app.api.store import InMemoryRunStore
from app.main import create_app


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _fake_request(store: InMemoryRunStore) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(run_service=SimpleNamespace(store=store))))


def _bearer(raw_key: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=raw_key)


def test_hash_api_key_is_deterministic():
    assert hash_api_key("my-key") == hash_api_key("my-key")


def test_hash_api_key_differs_by_salt(monkeypatch):
    monkeypatch.setattr(auth_module, "AUTH_API_KEY_SALT", "salt-a")
    hash_a = hash_api_key("my-key")
    monkeypatch.setattr(auth_module, "AUTH_API_KEY_SALT", "salt-b")
    hash_b = hash_api_key("my-key")
    assert hash_a != hash_b


def test_generate_api_key_produces_unique_values():
    assert generate_api_key() != generate_api_key()


@pytest.mark.anyio
async def test_require_api_key_raises_401_when_header_missing():
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(_fake_request(InMemoryRunStore()), credentials=None)
    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_require_api_key_raises_401_for_unknown_key():
    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(_fake_request(InMemoryRunStore()), credentials=_bearer("unknown"))
    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_require_api_key_raises_401_for_revoked_key():
    store = InMemoryRunStore()
    record = ApiKeyRecord(hashed_key=hash_api_key("raw"), label="test")
    await store.create_api_key(record)
    await store.revoke_api_key(record.id)

    with pytest.raises(HTTPException) as exc_info:
        await require_api_key(_fake_request(store), credentials=_bearer("raw"))
    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_require_api_key_returns_record_for_valid_key():
    store = InMemoryRunStore()
    record = ApiKeyRecord(hashed_key=hash_api_key("raw"), label="test")
    await store.create_api_key(record)

    result = await require_api_key(_fake_request(store), credentials=_bearer("raw"))
    assert result.id == record.id


@pytest.mark.anyio
async def test_require_admin_rejects_user_role():
    key = ApiKeyRecord(hashed_key="x", label="test", role=ApiKeyRole.USER)
    with pytest.raises(HTTPException) as exc_info:
        await require_admin(key)
    assert exc_info.value.status_code == 403


@pytest.mark.anyio
async def test_require_admin_allows_admin_role():
    key = ApiKeyRecord(hashed_key="x", label="test", role=ApiKeyRole.ADMIN)
    result = await require_admin(key)
    assert result is key


def test_is_authorized_owner_admin_sees_everything():
    admin = ApiKeyRecord(hashed_key="x", label="test", role=ApiKeyRole.ADMIN)
    assert is_authorized_owner(admin, None) is True
    assert is_authorized_owner(admin, uuid4()) is True


def test_is_authorized_owner_fails_closed_for_unowned_resource():
    user = ApiKeyRecord(hashed_key="x", label="test", role=ApiKeyRole.USER)
    assert is_authorized_owner(user, None) is False


def test_is_authorized_owner_matches_own_resource():
    user = ApiKeyRecord(hashed_key="x", label="test", role=ApiKeyRole.USER)
    assert is_authorized_owner(user, user.id) is True


def test_is_authorized_owner_rejects_other_owner():
    user = ApiKeyRecord(hashed_key="x", label="test", role=ApiKeyRole.USER)
    assert is_authorized_owner(user, uuid4()) is False


@pytest.mark.anyio
async def test_ensure_bootstrap_admin_key_noop_when_not_configured(monkeypatch):
    monkeypatch.setattr(auth_module, "AUTH_BOOTSTRAP_ADMIN_KEY", None)
    store = InMemoryRunStore()

    await ensure_bootstrap_admin_key(store)

    assert await store.list_api_keys() == []


@pytest.mark.anyio
async def test_ensure_bootstrap_admin_key_seeds_idempotently(monkeypatch):
    monkeypatch.setattr(auth_module, "AUTH_BOOTSTRAP_ADMIN_KEY", "bootstrap-secret")
    store = InMemoryRunStore()

    await ensure_bootstrap_admin_key(store)
    await ensure_bootstrap_admin_key(store)  # must not create a duplicate

    keys = await store.list_api_keys()
    assert len(keys) == 1
    assert keys[0].role == ApiKeyRole.ADMIN

    authenticated = await require_api_key(_fake_request(store), credentials=_bearer("bootstrap-secret"))
    assert authenticated.role == ApiKeyRole.ADMIN


@pytest.mark.anyio
async def test_admin_api_key_lifecycle_via_http():
    store = InMemoryRunStore()
    admin_record = ApiKeyRecord(hashed_key=hash_api_key("admin-raw"), label="admin", role=ApiKeyRole.ADMIN)
    await store.create_api_key(admin_record)
    admin_headers = {"Authorization": "Bearer admin-raw"}
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        created = await client.post(
            "/api/v1/admin/api-keys",
            json={"label": "new user", "role": "user"},
            headers=admin_headers,
        )
        assert created.status_code == 201
        body = created.json()
        assert "api_key" in body
        new_raw_key = body["api_key"]
        new_key_id = body["id"]

        listed = await client.get("/api/v1/admin/api-keys", headers=admin_headers)
        assert listed.status_code == 200
        assert any(k["id"] == new_key_id for k in listed.json())
        assert all("hashed_key" not in k and "api_key" not in k for k in listed.json())

        new_key_headers = {"Authorization": f"Bearer {new_raw_key}"}
        works_before_revoke = await client.post("/api/v1/threads", json={}, headers=new_key_headers)
        assert works_before_revoke.status_code == 201

        revoked = await client.delete(f"/api/v1/admin/api-keys/{new_key_id}", headers=admin_headers)
        assert revoked.status_code == 204

        fails_after_revoke = await client.post("/api/v1/threads", json={}, headers=new_key_headers)
        assert fails_after_revoke.status_code == 401


@pytest.mark.anyio
async def test_non_admin_cannot_manage_api_keys():
    store = InMemoryRunStore()
    user_record = ApiKeyRecord(hashed_key=hash_api_key("user-raw"), label="user", role=ApiKeyRole.USER)
    await store.create_api_key(user_record)
    user_headers = {"Authorization": "Bearer user-raw"}
    transport = httpx.ASGITransport(app=create_app(store))

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/api/v1/admin/api-keys", headers=user_headers)

    assert response.status_code == 403
