"""No real requests: permission, cache and Muse tool-round regressions."""
import pytest

from backends.meta import client


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.delenv("LIVESPLAT_ALLOW_META", raising=False)
    monkeypatch.setattr(client.cache, "get", lambda req: None)
    monkeypatch.setattr(client.cache, "put", lambda *a, **k: None)
    monkeypatch.setattr(client.budget, "record", lambda *a, **k: None)
    monkeypatch.setattr(client.budget, "check", lambda *a: None)
    def forbidden(*a, **k):
        pytest.fail("An unmocked HTTP request was attempted")
    monkeypatch.setattr(client.requests, "post", forbidden)


def test_cache_miss_never_calls_without_permission():
    with pytest.raises(client.PermissionRequired):
        client.sam31_image("bottle", "data:test")
    with pytest.raises(client.PermissionRequired):
        client.spark_chat([{"role": "user", "content": "hello"}])


def test_cache_hits_work_offline(monkeypatch):
    response = {"cached": True}
    monkeypatch.setattr(client.cache, "get", lambda req: response)
    assert client.sam31_image("bottle", "data:test") is response
    assert client.spark_chat([]) is response


def test_cache_only_wins_even_when_live_enabled(monkeypatch):
    monkeypatch.setenv("LIVESPLAT_ALLOW_META", "1")
    with pytest.raises(client.PermissionRequired):
        client.sam31_image("bottle", "data:test", cache_only=True)
    with pytest.raises(client.PermissionRequired):
        client.spark_chat([], cache_only=True)


def test_null_tool_content_and_selection_followup(monkeypatch):
    from types import SimpleNamespace
    from agent.planner import handle
    monkeypatch.setenv("LIVESPLAT_ALLOW_META", "1")
    monkeypatch.setattr(client, "_headers", lambda **kw: {})
    sent = []
    replies = [
        {"content": None, "tool_calls": [{"id": "a", "function": {"name": "list_objects", "arguments": "{}"}}]},
        {"content": None, "tool_calls": [{"id": "b", "function": {"name": "drop", "arguments": '{"obj":"1"}'}}]},
    ]
    def post(url, **kw):
        sent.append(kw["json"])
        body = {"choices": [{"message": replies[len(sent) - 1]}]}
        return SimpleNamespace(status_code=200, json=lambda: body)
    monkeypatch.setattr(client.requests, "post", post)
    ctrl = SimpleNamespace(names={1: ["bottle"]}, sc=SimpleNamespace(objects=[{"id": 1, "mass": .1}]),
                           w=SimpleNamespace(gravity=9.81), label=lambda k: "bottle", run=lambda c: "OK")
    done, note = handle("drop it", ctrl, "meta", default=1)
    assert [c["name"] for c, _ in done] == ["list_objects", "drop"]
    assert "Selected object: id 1" in sent[0]["messages"][0]["content"]


def test_sam_retry_rechecks_budget_before_request(monkeypatch):
    monkeypatch.setenv("LIVESPLAT_ALLOW_META", "1")
    monkeypatch.setattr(client, "_headers", lambda **kw: {})
    calls, checks, records = [], [], []
    def post(*a, **k):
        calls.append(1)
        raise client.requests.exceptions.Timeout()
    def check(*a):
        checks.append(1)
        if len(checks) == 2:
            raise client.budget.BudgetExceeded("limit")
    monkeypatch.setattr(client.requests, "post", post)
    monkeypatch.setattr(client.budget, "check", check)
    monkeypatch.setattr(client.budget, "record", lambda *a: records.append(a))
    with pytest.raises(client.budget.BudgetExceeded):
        client.sam31_image("bottle", "data:test")
    assert len(calls) == 1 and len(checks) == 2 and len(records) == 1
