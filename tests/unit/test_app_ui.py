"""Offline UI construction and callback tests; no models or server required."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import app


@pytest.fixture(autouse=True)
def isolated_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_EXPORT_DIR", SimpleNamespace(name=str(tmp_path)))


def test_demo_callback_is_explicitly_synthetic():
    status, review, rendered, paths, review_paths = app.show_demo()
    assert status.startswith("演示样例")
    assert "正式评测" not in status
    assert "模型概率" not in rendered
    assert review == app.DEMO_RESULT["review"]
    assert re.search(r'id="run-[0-9a-f]+-ev-1-1"', rendered)
    assert all(Path(path).is_file() for path in [*paths, *review_paths])
    assert json.loads(Path(paths[2]).read_text("utf-8"))["metadata"]["is_demo"] is True


def test_import_callback_and_failure_clear_previous_outputs(tmp_path):
    path = tmp_path / "raw.json"
    result = json.loads(json.dumps(app.DEMO_RESULT))
    result["sentences"][0]["evidences"][0]["corpus_verified"] = True
    path.write_text(json.dumps(result), "utf-8")
    success = app.import_result(path)
    assert "来源未核验" in success[0] and success[1]
    assert "未校验" in success[2] and "✓ 已通过" not in success[2]
    assert app.import_result(None)[1:] == ("", "", None, None)
    path.write_text("{", "utf-8")
    assert app.import_result(path)[1:] == ("", "", None, None)


def test_generation_success_clears_then_displays(monkeypatch):
    received = {}

    def run(*args, **kwargs):
        received.update(kwargs)
        return app.DEMO_RESULT

    monkeypatch.setattr(app, "run_pipeline", run)
    updates = list(app.generate_review("q", "paper.pdf", 3, 0.7, backend_python="chosen"))
    assert len(updates) == 2
    assert updates[0][1:] == ("", "", None, None)
    assert updates[1][1] == app.DEMO_RESULT["review"]
    assert "未校验" in updates[1][2]
    assert received["backend_python"] == "chosen"


def test_generation_automatically_checks_each_evidence(monkeypatch):
    monkeypatch.setattr(app, "run_pipeline", lambda *args, **kwargs: app.DEMO_RESULT)
    example = app.DEMO_RESULT["sentences"][0]["evidences"][0]
    index = {(example["source"], example["para_id"]): example}
    *_, generated = app.generate_review("q", "paper.pdf", 3, 0.7, corpus_index=index)
    assert "✓ 已通过语料一致性校验" in generated[2]
    assert "⚠ 与冻结语料不一致" in generated[2]
    envelope = json.loads(Path(generated[3][2]).read_text("utf-8"))
    evidence = envelope["result"]["sentences"][0]["evidences"]
    assert [item["corpus_verified"] for item in evidence] == [True, False]


def test_generation_failure_clears_stale_outputs(monkeypatch):
    app.show_demo()

    def fail(*args, **kwargs):
        raise ValueError("后端不可用")

    monkeypatch.setattr(app, "run_pipeline", fail)
    updates = list(app.generate_review("q", "paper.pdf", 3, 0.7))
    assert len(updates) == 2
    assert all(update[1:] == ("", "", None, None) for update in updates)
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
            assert len(functions[name].outputs) == 5
        assert functions["generate_review"].inputs[1].file_count == "multiple"
        assert functions["clear_result"].fn()[1:] == ("", "", None, None)
        assert functions["show_demo"].fn()[0].startswith("演示样例")
        diagnostics = functions["check_environment"]
        assert diagnostics.concurrency_id == "review-actions"
        assert diagnostics.concurrency_limit == 1
        assert len(diagnostics.outputs) == 1
        assert diagnostics.outputs[0] not in functions["show_demo"].outputs
        corpus = functions["load_frozen_corpus"]
        assert len(corpus.inputs) == 1 and len(corpus.outputs) == 2
        assert corpus.inputs[0].file_count == "single"
        assert corpus.outputs[0] is functions["generate_review"].inputs[-1]
        assert isinstance(corpus.outputs[0], gr.State)
        assert corpus.outputs[1].interactive is False
        assert corpus.queue is True
        assert corpus.concurrency_id == "review-actions"
        assert corpus.concurrency_limit == 1
        selected = []
        monkeypatch.setattr(
            app, "check_environment", lambda python: selected.append(python) or "自检结果"
        )
        assert diagnostics.fn() == "自检结果"
        assert selected == ["chosen"]
    finally:
        demo.close()


def test_reloading_result_does_not_reuse_previous_selection_anchor():
    first = app.show_demo()[2]
    second = app.show_demo()[2]
    first_ids = set(re.findall(r'id="([^"]+)"', first))
    second_ids = set(re.findall(r'id="([^"]+)"', second))
    assert first_ids and second_ids and first_ids.isdisjoint(second_ids)
    assert set(re.findall(r'href="#([^"]+)"', first)) <= first_ids
    assert set(re.findall(r'href="#([^"]+)"', second)) <= second_ids


def test_workspace_has_guidance_and_responsive_shell_without_export_css(tmp_path):
    assert "加载演示样例" in app.EMPTY_WORKSPACE
    assert "<h2>综述</h2>" in app.EMPTY_WORKSPACE
    assert "<h2>引用证据</h2>" in app.EMPTY_WORKSPACE
    assert "max-width:none" in app.UI_CSS
    assert "#workspace-layout {grid-template-columns:minmax(0, 1fr);}" in app.UI_CSS
    assert "grid-template-columns:minmax(0, 2fr) minmax(0, 3fr)" in app.UI_CSS
    assert "#output-panel" not in app.CSS
    paths = app.export_report(app.DEMO_RESULT, is_demo=True)
    exported = next(p for p in paths if str(p).endswith(".html"))
    assert "#output-panel" not in Path(exported).read_text()


def test_advanced_tools_are_folded_away_from_main_flow():
    demo = app.build_app()
    try:
        config = demo.get_config_file()
        components = {item["id"]: item for item in config["components"]}
        advanced = next(
            item for item in components.values() if item["props"].get("label") == "高级"
        )
        assert advanced["props"]["open"] is False

        def element(elem_id):
            return next(
                item["id"]
                for item in components.values()
                if item["props"].get("elem_id") == elem_id
            )

        workspace = next(
            child
            for child in config["layout"]["children"]
            if child["id"] == element("workspace-layout")
        )
        inputs, output = workspace["children"]
        assert inputs["id"] == element("input-panel")
        assert output["id"] == element("output-panel")
        children = next(
            child["children"] for child in inputs["children"] if child["id"] == advanced["id"]
        )
        labels = {components[child["id"]]["props"].get("label") for child in children}
        assert {"证据溯源报告", "结果导入", "冻结语料", "检索与匹配设置"} <= labels
        assert "生成环境自检" not in labels
        copy = " ".join(
            item["props"].get("value", "")
            for item in components.values()
            if item["type"] in {"markdown", "html"}
        )
        assert "MiB" not in copy
        assert "句级匹配仅提供证据支持" not in copy
        assert any(
            item["props"].get("label") == "下载综述"
            and item["id"] not in {child["id"] for child in children}
            for item in components.values()
        )
    finally:
        demo.close()


def test_workspace_preserves_content_that_matches_ui_labels():
    # These strings used to be replaced across the entire rendered document.
    result = json.loads(json.dumps(app.DEMO_RESULT))
    text = "原文证据 · 文献与位置 [证据1: 文献甲 · 第1页 第1段] <script>例子</script>"
    result["query"] = text
    result["review"] = text
    result["sentences"][0]["sentence"] = text
    result["sentences"][0]["evidences"][0].update(source="文献甲", text=text)
    rendered = app.render_workspace(result, anchor_prefix="preserve-")
    import html

    assert rendered.count(html.escape(text)) == 3  # query, sentence and excerpt
    assert "<script>" not in rendered
    assert 'href="#preserve-ev-1-1">证据 1 ↗</a>' in rendered
    assert "<h3>引用证据</h3>" in rendered


def test_partial_sentence_result_keeps_complete_review_accessible():
    import gradio as gr

    result = {
        "query": "研究问题",
        "review": "已匹配的正文。\n仍未进入句级结果的正文。",
        "sentences": [{"sentence": "已匹配的正文。", "evidences": []}],
    }
    outputs = app._display(result, "测试导入")
    assert outputs[1] == result["review"]
    demo = app.build_app()
    try:
        show_demo = next(fn for fn in demo.fns.values() if fn.api_name == "show_demo")
        full_review = show_demo.outputs[1]
        assert full_review.visible is True  # Not an inaccessible hidden API output.
        assert full_review.show_copy_button is True
        parent = full_review.parent
        while parent is not None and not isinstance(parent, gr.Accordion):
            assert parent.visible is True
            parent = parent.parent
        assert isinstance(parent, gr.Accordion)
        assert parent.visible is True and parent.open is False
        assert parent.label == "综述全文"
    finally:
        demo.close()


def test_workspace_without_sentences_displays_full_review_and_warning():
    result = {"query": "研究问题", "review": "未切分的完整正文。", "sentences": []}
    rendered = app.render_workspace(result, anchor_prefix="empty-")
    assert result["review"] in rendered
    assert "没有逐句结果" in rendered


@pytest.mark.parametrize("is_demo", [False, True])
def test_review_downloads_include_full_text_and_no_evidence_report(tmp_path, is_demo):
    from docx import Document

    result = {
        **app.DEMO_RESULT,
        "query": "文献讨论了什么？",
        "review": "第一段 <原文>。\n同段换行。\n\n完整的第二段。",
    }
    markdown_path, docx_path = map(Path, app.export_review(result, tmp_path, is_demo=is_demo))
    markdown = markdown_path.read_text("utf-8")
    document = Document(docx_path)
    assert markdown_path.name == "review.md"
    assert docx_path.name == "review.docx"
    assert "第一段 &lt;原文&gt;。<br>同段换行。\n\n完整的第二段。" in markdown
    assert "## 逐句证据对照" not in markdown
    assert (app.DEMO_NOTICE in markdown) is is_demo
    assert [paragraph.text for paragraph in document.paragraphs] == [
        "文献综述",
        *([app.DEMO_NOTICE] if is_demo else []),
        "研究问题",
        result["query"],
        "综述正文",
        *result["review"].split("\n\n"),
    ]
    assert document.styles["Normal"].paragraph_format.line_spacing == 1.7
