from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from smart_router.engine.engine import EngineWorkersResponse, RequestType
from smart_router.entrypoints.serve.vllm_routes import VllmRoutes


class FakeResponse:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class FakeHttpClient:
    def __init__(self, responses: dict[str, FakeResponse]):
        self._responses = responses

    async def get(self, url: str, headers=None):
        return self._responses[url]

    async def aclose(self):
        return None


class FakeEngineClient:
    identity = "models-test"

    def __init__(self, urls):
        self.urls = urls

    async def send_request(self, request):
        assert request.request_type == RequestType.WORKERS
        import asyncio

        fut = asyncio.get_running_loop().create_future()
        fut.set_result(
            EngineWorkersResponse(request_id=request.request_id, urls=self.urls)
        )
        return fut


def test_models_route_aggregates_and_deduplicates_upstream_models():
    routes = VllmRoutes(
        http_client=FakeHttpClient(
            {
                "http://prefill-a/v1/models": FakeResponse(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "model-a", "object": "model", "owned_by": "worker-a"},
                        ],
                    },
                ),
                "http://decode-b/v1/models": FakeResponse(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {"id": "model-a", "max_model_len": 32768},
                            {"id": "model-b", "max_model_len": 8192},
                        ],
                    },
                ),
            }
        )
    )
    app = Starlette(routes=[Route("/v1/models", routes.models, methods=["GET"])])
    app.state.model_source_urls = ["http://prefill-a", "http://decode-b"]

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "list"
    assert [model["id"] for model in body["data"]] == ["model-a", "model-b"]
    assert body["data"][0]["owned_by"] == "worker-a"
    assert body["data"][0]["max_model_len"] == 32768
    assert body["data"][1]["max_model_len"] == 8192


def test_models_route_returns_503_when_all_upstreams_fail():
    routes = VllmRoutes(
        http_client=FakeHttpClient(
            {
                "http://prefill-a/v1/models": FakeResponse(500, {"error": "boom"}),
                "http://decode-b/v1/models": FakeResponse(404, {"error": "missing"}),
            }
        )
    )
    app = Starlette(routes=[Route("/v1/models", routes.models, methods=["GET"])])
    app.state.model_source_urls = ["http://prefill-a", "http://decode-b"]

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 503
    assert response.json()["error"] == "No available upstream /v1/models endpoint"


def test_models_route_uses_dynamic_worker_urls_from_engine():
    routes = VllmRoutes(
        http_client=FakeHttpClient(
            {
                "http://dynamic-prefill/v1/models": FakeResponse(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": "dynamic-model"}],
                    },
                ),
            }
        )
    )
    app = Starlette(routes=[Route("/v1/models", routes.models, methods=["GET"])])
    app.state.model_source_urls = []
    app.state.engine_client = FakeEngineClient(["http://dynamic-prefill"])

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "dynamic-model"
