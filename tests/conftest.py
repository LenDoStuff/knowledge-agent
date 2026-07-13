import pytest
from pydantic_ai import models


LIVE_MARKERS = ("live_nvidia", "live_azure", "live_api_key_ingestion")


@pytest.fixture(autouse=True)
def block_unexpected_model_requests(request):
    previous = models.ALLOW_MODEL_REQUESTS
    is_live = any(request.node.get_closest_marker(name) for name in LIVE_MARKERS)
    models.ALLOW_MODEL_REQUESTS = is_live
    try:
        yield
    finally:
        models.ALLOW_MODEL_REQUESTS = previous
