import importlib
import sys
import types

import pytest


@pytest.fixture()
def upload_to_modelscope(monkeypatch):
    modelscope = types.ModuleType("modelscope")
    hub = types.ModuleType("modelscope.hub")
    api = types.ModuleType("modelscope.hub.api")
    api.HubApi = object
    monkeypatch.setitem(sys.modules, "modelscope", modelscope)
    monkeypatch.setitem(sys.modules, "modelscope.hub", hub)
    monkeypatch.setitem(sys.modules, "modelscope.hub.api", api)
    module = importlib.import_module("scripts.upload_to_modelscope")
    return importlib.reload(module)


def test_get_env_value_prefers_first_exported_name(upload_to_modelscope, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_ACCESS_TOKEN", "access-token")
    monkeypatch.setenv("MODELSCOPE_TOKEN", "preferred-token")

    value = upload_to_modelscope.get_env_value(
        "MODELSCOPE_TOKEN",
        "MODELSCOPE_ACCESS_TOKEN",
        default="fallback-token",
    )

    assert value == "preferred-token"


def test_get_env_value_falls_back_to_default(upload_to_modelscope, monkeypatch):
    monkeypatch.delenv("MODELSCOPE_TOKEN", raising=False)
    monkeypatch.delenv("MODELSCOPE_ACCESS_TOKEN", raising=False)

    value = upload_to_modelscope.get_env_value(
        "MODELSCOPE_TOKEN",
        "MODELSCOPE_ACCESS_TOKEN",
        default="fallback-token",
    )

    assert value == "fallback-token"


def test_get_env_int_rejects_invalid_values(upload_to_modelscope, monkeypatch):
    monkeypatch.setenv("MODELSCOPE_VISIBILITY", "private")

    try:
        upload_to_modelscope.get_env_int("MODELSCOPE_VISIBILITY", default=1)
    except ValueError as exc:
        assert "MODELSCOPE_VISIBILITY" in str(exc)
    else:
        raise AssertionError("Expected invalid integer env var to raise ValueError")
