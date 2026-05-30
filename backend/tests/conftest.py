import os
import tempfile
import pytest

# Phase A bootstrap tests do not need FastAPI / DB
_BOOTSTRAP_ONLY = os.environ.get("SHIPAI_BOOTSTRAP_ONLY") == "1"

if not _BOOTSTRAP_ONLY:
    # Set the test data directory BEFORE importing database or app modules
    TEST_DIR = tempfile.mkdtemp()
    os.environ["SHIPAI_DATA_DIR"] = TEST_DIR

    from httpx import AsyncClient
    from app.core.database import DB_PATH
    from app.main import app

    @pytest.fixture(scope="session", autouse=True)
    def setup_test_db():
        yield
        import shutil
        try:
            shutil.rmtree(TEST_DIR)
        except Exception:
            pass

    import pytest_asyncio

    @pytest_asyncio.fixture
    async def async_client():
        from app.core.database import init_db

        await init_db()
        from httpx import ASGITransport

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            yield client
