"""HTTP contracts relied on by existing parser callers, without model or broker I/O.

Tier defaults and durable job identity/recovery are covered by their dedicated
tests. These regressions exercise field locations and the public JSON boundary.
"""

import importlib
import runpy
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.services import job_store, job_submission

ROUTES = (
    ("/mineru", "mineru_router"),
    ("/mineru_sci", "mineru_sci_router"),
    ("/mineru_with_images", "mineru_with_images_router"),
    ("/mineru/task", "mineru_task_router"),
    ("/mineru_with_images/task", "mineru_with_images_task_router"),
    ("/two_stage/task", "two_stage_router"),
)
SYNC_ROUTES = ROUTES[:3]
TASK_ROUTES = ROUTES[3:]
UPLOAD = {"file": ("sample.pdf", b"%PDF-legacy-upload", "application/pdf")}
CHUNKS = [
    {"text": "研究标题", "page_number": 3, "type": "title"},
    {"text": "页眉", "page_number": 3, "type": "header"},
    {"text": "结论 ☑\n正文", "page_number": 3, "type": None},
    {"text": "<table><tr><td>42%</td></tr></table>", "page_number": 3},
    {"text": "图示描述", "page_number": 3, "type": "image"},
    {"text": "3", "page_number": 3, "type": "page_number"},
    {"text": "页脚", "page_number": 3, "type": "footer"},
    {"text": "下一源页", "page_number": 11, "type": "header"},
    {"text": "最后正文", "page_number": 11},
]


@pytest.fixture
def dispatches(client, monkeypatch, route):
    """Keep upload persistence real; replace only parser and broker boundaries."""
    module = importlib.import_module(f"src.routers.{route[1]}")
    captured = []

    def capture(source, options):
        assert Path(source).read_bytes() == UPLOAD["file"][1]
        captured.append(options)

    if route in SYNC_ROUTES:

        def submit(source, **options):
            capture(source, options)
            future = Future()
            future.set_result({"result": CHUNKS, "txt": "unfiltered worker TXT"})
            return future

        monkeypatch.setattr(module.scheduler, "submit", submit)
    else:

        def publish(record):
            capture(job_store.source_path(record["job_id"]), record["options"])

        monkeypatch.setattr(job_submission, "publish_record", publish)
    return captured


@pytest.mark.parametrize("route", ROUTES, ids=[route[0] for route in ROUTES])
@pytest.mark.parametrize("selected", [False, True])
def test_flags_use_legacy_query_or_form_even_when_other_location_conflicts(
    client, route, dispatches, selected
):
    selected_fields = {"chunk_type": str(selected).lower(), "return_txt": str(selected).lower()}
    conflicting_fields = {
        "chunk_type": str(not selected).lower(),
        "return_txt": str(not selected).lower(),
    }
    query, form = selected_fields, conflicting_fields
    if route in TASK_ROUTES[-1:]:
        query, form = form, query
    response = client.post(route[0], files=UPLOAD, params=query, data=form)
    assert response.status_code == 200, response.text
    assert len(dispatches) == 1
    assert dispatches[0]["chunk_type"] is selected
    assert dispatches[0]["return_txt"] is selected


@pytest.mark.parametrize("route", ROUTES, ids=[route[0] for route in ROUTES])
def test_upload_must_use_required_file_field(client, route, dispatches):
    response = client.post(route[0], files={"document": UPLOAD["file"]})
    assert response.status_code == 422
    assert any(error["loc"] == ["body", "file"] for error in response.json()["detail"])
    assert dispatches == []


@pytest.mark.parametrize("route", SYNC_ROUTES, ids=[route[0] for route in SYNC_ROUTES])
@pytest.mark.parametrize("chunk_type", [False, True])
@pytest.mark.parametrize("return_txt", [False, True])
def test_sync_result_keeps_source_pages_reading_order_and_txt_contract(
    client, route, dispatches, chunk_type, return_txt
):
    response = client.post(
        route[0],
        files=UPLOAD,
        params={"chunk_type": str(chunk_type).lower(), "return_txt": str(return_txt).lower()},
    )
    assert response.status_code == 200, response.text
    if chunk_type:
        expected = [
            {"text": "研究标题", "page_number": 3, "type": "title"},
            {"text": "页眉", "page_number": 3, "type": "header"},
            {"text": "结论 ☑\n正文", "page_number": 3},
            {"text": "<table><tr><td>42%</td></tr></table>", "page_number": 3},
            {"text": "图示描述", "page_number": 3, "type": "image"},
            {"text": "页脚", "page_number": 3, "type": "footer"},
            {"text": "下一源页", "page_number": 11, "type": "header"},
            {"text": "最后正文", "page_number": 11},
        ]
        txt = (
            "研究标题\n\n页眉\n结论 ☑\n正文\n<table><tr><td>42%</td></tr></table>\n"
            "图示描述\n页脚\n下一源页\n最后正文"
        )
    else:
        expected = [
            {"text": "研究标题", "page_number": 3},
            {"text": "结论 ☑\n正文", "page_number": 3},
            {"text": "<table><tr><td>42%</td></tr></table>", "page_number": 3},
            {"text": "图示描述", "page_number": 3},
            {"text": "最后正文", "page_number": 11},
        ]
        txt = "研究标题\n结论 ☑\n正文\n<table><tr><td>42%</td></tr></table>\n图示描述\n最后正文"
    payload = {"result": expected}
    if return_txt:
        payload["txt"] = txt
    assert response.json() == payload


@pytest.fixture
def authenticated_client(client, monkeypatch):
    """Build the real main entrypoint with auth enabled without reloading shared modules."""
    config = importlib.import_module("src.config.config")
    monkeypatch.setattr(config, "FASTAPI_AUTH", True)
    monkeypatch.setattr(config, "FASTAPI_BEARER_TOKEN", "legacy-test-token")
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "src" / "main.py"))
    # Do not run lifespan: it would shut down the session fixture's scheduler.
    return TestClient(namespace["app"])


@pytest.mark.parametrize("route", ROUTES, ids=[route[0] for route in ROUTES])
def test_all_six_upload_routes_enforce_bearer_auth_before_dispatch(
    authenticated_client, route, dispatches
):
    for headers in ({}, {"Authorization": "Bearer invalid-test-token"}):
        response = authenticated_client.post(route[0], files=UPLOAD, headers=headers)
        assert response.status_code == 401
    assert dispatches == []
    accepted = authenticated_client.post(
        route[0], files=UPLOAD, headers={"Authorization": "Bearer legacy-test-token"}
    )
    assert accepted.status_code == 200, accepted.text
    assert len(dispatches) == 1


@pytest.mark.parametrize("route", TASK_ROUTES, ids=[route[0] for route in TASK_ROUTES])
@pytest.mark.parametrize("return_txt", [False, True])
def test_pre_durable_task_success_keeps_nested_result_and_mode_specific_nulls(
    client, monkeypatch, route, return_txt
):
    module = importlib.import_module(f"src.routers.{route[1]}")
    chunks = [
        {"text": "保留正文", "page_number": 3, "type": None},
        {"text": "图示描述", "page_number": 3, "type": "image"},
        {"text": "研究标题", "page_number": 11, "type": "title"},
    ]
    txt = "保留正文\n图示描述\n研究标题" if return_txt else None
    monkeypatch.setattr(
        module,
        "AsyncResult",
        lambda task_id, **kwargs: SimpleNamespace(
            state="SUCCESS", result={"result": chunks, "txt": txt}
        ),
    )
    response = client.get(f"{route[0]}/legacy-task-id")
    assert response.status_code == 200
    expected_chunks = [dict(chunk) for chunk in chunks]
    payload = {"result": expected_chunks}
    if route[0] == "/two_stage/task":
        payload["txt"] = txt
    else:
        del expected_chunks[0]["type"]
        if return_txt:
            payload["txt"] = txt
    assert response.json() == {"task_id": "legacy-task-id", "state": "SUCCESS", "result": payload}


@pytest.mark.parametrize("route", TASK_ROUTES, ids=[route[0] for route in TASK_ROUTES])
@pytest.mark.parametrize("state", ["FAILURE", "REVOKED"])
def test_pre_durable_task_terminal_failure_keeps_legacy_http_status(
    client, monkeypatch, route, state
):
    module = importlib.import_module(f"src.routers.{route[1]}")
    monkeypatch.setattr(
        module,
        "AsyncResult",
        lambda task_id, **kwargs: SimpleNamespace(state=state, info=RuntimeError("parse failed")),
    )
    response = client.get(f"{route[0]}/legacy-task-id")
    assert response.status_code == (200 if route[0] == "/two_stage/task" else 500)
    assert response.json() == {"task_id": "legacy-task-id", "state": state, "error": "parse failed"}
