import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_root_api(async_client: AsyncClient):
    response = await async_client.get("/api")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert data["name"] == "ShipAI"

async def test_auth_registration(async_client: AsyncClient):
    # Test Register
    response = await async_client.post(
        "/api/auth/register",
        json={"email": "test@shipai.com", "password": "securepassword", "display_name": "Test User"}
    )
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["email"] == "test@shipai.com"
    assert "id" in data
    
    # Test Duplicate Registration
    response2 = await async_client.post(
        "/api/auth/register",
        json={"email": "test@shipai.com", "password": "securepassword"}
    )
    assert response2.status_code == 400

async def test_auth_login(async_client: AsyncClient):
    # Make sure user exists
    await async_client.post(
        "/api/auth/register",
        json={"email": "login@shipai.com", "password": "securepassword"}
    )
    
    # Test Login
    response = await async_client.post(
        "/api/auth/login",
        data={"username": "login@shipai.com", "password": "securepassword"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    
    # Test /me
    token = data["access_token"]
    me_response = await async_client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "login@shipai.com"

async def test_project_management(async_client: AsyncClient):
    # Register and Login
    await async_client.post(
        "/api/auth/register",
        json={"email": "builder@shipai.com", "password": "securepassword"}
    )
    login_res = await async_client.post(
        "/api/auth/login",
        data={"username": "builder@shipai.com", "password": "securepassword"}
    )
    token = login_res.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # Create Project
    create_res = await async_client.post(
        "/api/projects/create",
        headers=headers,
        json={"name": "My AI Agent", "template": "multi_agent", "config_json": "{}"}
    )
    assert create_res.status_code == 200
    project_id = create_res.json()["project"]["id"]
    
    # List Projects
    list_res = await async_client.get("/api/projects/", headers=headers)
    assert list_res.status_code == 200
    assert len(list_res.json()) >= 1
    
    # Delete Project
    del_res = await async_client.delete(f"/api/projects/{project_id}", headers=headers)
    assert del_res.status_code == 200
