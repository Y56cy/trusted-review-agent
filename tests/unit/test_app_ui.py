"""Offline UI construction and callback tests; no models or server required."""

import json
from pathlib import Path

import pytest

import app


@pytest.fixture(autouse=True)
def isolated_exports(tmp_path, monkeypatch):
    original = app.export_report

    def export(result, **kwargs):
        return original(result, tmp_path, **kwargs)

    monkeypatch.setattr(app, "export_report", export)


def test_demo_callback_is_explicitly_synthetic():
    status, review, rendered, paths = app.show_demo()
    assert app.DEMO_NOTICE in status
    assert "不是正式评测" in status
    assert review == app.DEMO_RESULT["review"]
    assert 'id="ev-1-1"' in rendered
    assert all(Path(path).is_file() for path in paths)
    assert json.loads(Path(paths[2]).read_text("utf-8"))["metadata"]["is_demo"] is True


def test_import_callback_and_failure_clear_previous_outputs(tmp_path):
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(app.DEMO_RESULT), "utf-8")
    success = app.import_result(path)
    assert "来源未核验" in success[0] and success[1]
    assert app.import_result(None)[1:] == ("", "", None)
    path.write_text("{", "utf-8")
    assert app.import_result(path)[1:] == ("", "", None)


def test_generation_success_clears_then_displays(monkeypatch):
    received = {}

    def run(*args, **kwargs):
        received.update(kwargs)
        return app.DEMO_RESULT

    monkeypatch.setattr(app, "run_pipeline", run)
    updates = list(app.generate_review("q", "paper.pdf", 3, 0.7, backend_python="chosen"))
    assert len(updates) == 2
    assert updates[0][1:] == ("", "", None)
    assert updates[1][1] == app.DEMO_RESULT["review"]
    assert received["backend_python"] == "chosen"


def test_generation_failure_clears_stale_outputs(monkeypatch):
    app.show_demo()

    def fail(*args, **kwargs):
        raise ValueError("后端不可用")

    monkeypatch.setattr(app, "run_pipeline", fail)
    updates = list(app.generate_review("q", "paper.pdf", 3, 0.7))
    assert len(updates) == 2
    assert all(update[1:] == ("", "", None) for update in updates)
    assert "后端不可用" in updates[-1][0]


def test_build_blocks_wires_callbacks_and_serializes_actions(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    import gradio as gr

    demo = app.build_app(backend_python="chosen")
    try:
        assert isinstance(demo, gr.Blocks)
        functions = {fn.api_name: fn for fn in demo.fns.values()}
        names = {"generate_review", "import_result", "show_demo", "clear_result"}
        assert names <= functions.keys()
        for name in names:
            assert functions[name].concurrency_id == "review-actions"
            assert functions[name].concurrency_limit == 1
            assert len(functions[name].outputs) == 4
        assert functions["clear_result"].fn()[1:] == ("", "", None)
        assert "手工构造" in functions["show_demo"].fn()[0]
    finally:
        demo.close()
