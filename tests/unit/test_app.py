"""Offline contract, safety, export and subprocess adapter regressions."""

import copy
import csv
import html
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

import app


@pytest.fixture
def result():
    return copy.deepcopy(app.DEMO_RESULT)


@pytest.fixture
def pdf(tmp_path):
    path = tmp_path / "论文.pdf"
    path.write_bytes(b"%PDF-1.7\nfixture: not a real model input")
    return path


def test_validate_returns_independent_canonical_copy(result):
    result["ignored"] = "extra"
    checked = app.validate_result(result)
    assert set(checked) == {"query", "review", "sentences"}
    checked["sentences"][0]["evidences"][0]["text"] = "changed"
    assert result["sentences"][0]["evidences"][0]["text"] != "changed"


@pytest.mark.parametrize("bad", [None, [], "text", 42, {}, {"query": "q"}])
def test_invalid_result_roots(bad):
    with pytest.raises(ValueError):
        app.validate_result(bad)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("query", None),
        ("review", 1),
        ("sentences", None),
        ("sentences", {}),
        ("sentences", [None]),
        ("sentences", [{"sentence": "s", "evidences": None}]),
        ("sentences", [{"sentence": 3, "evidences": []}]),
        ("sentences", [{"sentence": "s", "evidences": [None]}]),
    ],
)
def test_invalid_result_fields(result, key, bad):
    result[key] = bad
    with pytest.raises(ValueError):
        app.validate_result(result)


@pytest.mark.parametrize(
    "key,bad",
    [
        ("page", True),
        ("page", 0),
        ("page", -1),
        ("page", "1"),
        ("para_id", False),
        ("para_id", 1.5),
        ("para_id", None),
        ("text", None),
        ("nli_label", "unknown"),
        ("nli_label", []),
        ("nli_prob", True),
        ("nli_prob", "0.7"),
        ("nli_prob", -0.01),
        ("nli_prob", 1.01),
        ("nli_prob", float("nan")),
        ("nli_prob", float("inf")),
        ("nli_prob", float("-inf")),
        ("nli_prob", 10**400),
    ],
)
def test_invalid_evidence_fields(result, key, bad):
    result["sentences"][0]["evidences"][0][key] = bad
    with pytest.raises(ValueError):
        app.validate_result(result)


@pytest.mark.parametrize("prob", [0, 1, 0.7])
@pytest.mark.parametrize("label", ["entailment", "neutral", "contradiction"])
def test_valid_probability_and_label_boundaries(result, prob, label):
    evidence = result["sentences"][0]["evidences"][0]
    evidence.update(nli_prob=prob, nli_label=label)
    assert app.validate_result(result)["sentences"][0]["evidences"][0]["nli_prob"] == prob


def test_empty_results_warn_and_export(tmp_path):
    empty = {"query": "", "review": "", "sentences": []}
    assert app.validate_result(empty) == empty
    assert "没有逐句结果" in app.render_provenance(empty)
    assert len(app.export_report(empty, tmp_path)) == 4


def test_render_escapes_untrusted_text_and_internal_ids(result):
    payload = '<img src=x onerror="alert(1)"><script>alert(2)</script>&'
    result["sentences"][0]["sentence"] = payload
    result["sentences"][0]["evidences"][0]["text"] = payload
    rendered = app.render_provenance(result)
    assert payload not in rendered
    assert html.escape(payload) in rendered
    assert "<script>" not in rendered and "<img" not in rendered
    assert 'href="#ev-1-1"' in rendered and 'id="ev-1-1"' in rendered
    assert "未匹配到证据" in rendered


def test_exports_preserve_data_and_warn_without_fake_metrics(result, tmp_path):
    paths = {
        Path(p).name: Path(p)
        for p in app.export_report(
            result,
            tmp_path,
            source_name="演示来源",
            is_demo=True,
        )
    }
    assert set(paths) == {"report.md", "evidence.csv", "result.json", "report.html"}
    envelope = json.loads(paths["result.json"].read_text("utf-8"))
    assert envelope["schema_version"] == 1
    assert envelope["result"] == result
    assert envelope["metadata"]["is_demo"] is True
    assert "未进行正式评测" in envelope["metadata"]["evaluation_status"]
    for name in ("report.md", "report.html"):
        content = paths[name].read_text("utf-8")
        assert app.NOTICE in content
        assert "手工构造" in content
        assert "未进行正式评测" in content
    with paths["evidence.csv"].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    assert len(rows) == 3
    assert rows[1][4] == result["sentences"][0]["evidences"][0]["text"]
    assert "未匹配到证据" in rows[2][4]


@pytest.mark.parametrize(
    "payload", ['=HYPERLINK("https://invalid")', "+1", "-2", "@SUM(A1)", "  =1", "\t=1", "\r=1"]
)
def test_csv_formula_neutralization_and_json_original(result, tmp_path, payload):
    result["sentences"][0]["sentence"] = payload
    result["sentences"][0]["evidences"][0]["text"] = payload
    paths = app.export_report(result, tmp_path, source_name=payload)
    with Path(paths[1]).open(encoding="utf-8-sig", newline="") as stream:
        row = list(csv.reader(stream))[1]
    assert row[1] == row[4] == row[7] == "'" + payload
    assert json.loads(Path(paths[2]).read_text("utf-8"))["result"] == result


def test_html_export_escapes_query_review_source(result, tmp_path):
    payload = '<script>alert("test")</script>'
    result.update(query=payload, review=payload)
    paths = app.export_report(result, tmp_path, source_name=payload)
    content = Path(paths[3]).read_text("utf-8")
    assert payload not in content
    assert content.count(html.escape(payload)) == 3


def test_concurrent_exports_are_isolated(result, tmp_path):
    def export(index):
        current = copy.deepcopy(result)
        current["query"] = str(index)
        paths = app.export_report(current, tmp_path)
        return Path(paths[2]), current

    with ThreadPoolExecutor(max_workers=8) as pool:
        exports = list(pool.map(export, range(16)))
    assert len({path.parent for path, _ in exports}) == 16
    for path, expected in exports:
        assert json.loads(path.read_text("utf-8"))["result"] == expected


def test_load_raw_and_envelope_untrusted_origin(result, tmp_path):
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(result, ensure_ascii=False), "utf-8-sig")
    loaded, metadata = app.load_result(raw)
    assert loaded == result and "来源未核验" in metadata["source_name"]
    exported = app.export_report(result, tmp_path, is_demo=True, source_name="known")
    loaded, metadata = app.load_result(exported[2])
    assert loaded == result and metadata["is_demo"] is True
    assert "来源未核验" in metadata["source_name"]


@pytest.mark.parametrize(
    "data",
    [
        b"{",
        b"\xff",
        b"[]",
        b"null",
        b'{"value": NaN}',
        b'{"schema_version": true}',
        b'{"schema_version": 2}',
        b'{"schema_version":1,"metadata":{"is_demo":"false"}}',
    ],
)
def test_load_invalid_json(data, tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(data)
    with pytest.raises(ValueError):
        app.load_result(path)


def test_load_size_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MAX_JSON_BYTES", 3)
    path = tmp_path / "large.json"
    path.write_text("1234", "utf-8")
    with pytest.raises(ValueError, match="MiB"):
        app.load_result(path)


def test_fresh_import_has_no_ui_or_model_dependencies():
    script = """
import sys
class DenyHeavy:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'gradio', 'torch', 'transformers',
                                    'pipeline', 'sentence_transformers', 'match'}:
            raise AssertionError('unexpected import: ' + fullname)
sys.meta_path.insert(0, DenyHeavy())
import app
app.validate_result(app.DEMO_RESULT)
app.render_provenance(app.DEMO_RESULT)
import tempfile
with tempfile.TemporaryDirectory() as folder:
    app.export_report(app.DEMO_RESULT, folder)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(app.SRC)},
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_pipeline_adapter_sets_offline_flags_cwd_and_timeout(result, pdf, monkeypatch):
    def run(command, **kwargs):
        assert kwargs["cwd"] == app.SRC
        assert kwargs["timeout"] == 12
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert command[0] == "/chosen/python"
        request = json.loads(Path(command[-2]).read_text("utf-8"))
        assert request == {"query": "问题", "pdf_path": str(pdf), "top_k": 4, "nli_threshold": 0.8}
        Path(command[-1]).write_text(json.dumps(result), "utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(app.subprocess, "run", run)
    assert (
        app.run_pipeline(" 问题 ", pdf, 4.0, 0.8, timeout=12, backend_python="/chosen/python")
        == result
    )


@pytest.mark.parametrize(
    "query,k,threshold",
    [
        ("", 3, 0.7),
        ("  ", 3, 0.7),
        ("q", True, 0.7),
        ("q", 0, 0.7),
        ("q", 21, 0.7),
        ("q", 1.5, 0.7),
        ("q", 3, True),
        ("q", 3, float("nan")),
        ("q", 3, 1.1),
    ],
)
def test_invalid_request_never_starts_backend(query, k, threshold, pdf, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input must not start the backend")

    monkeypatch.setattr(app.subprocess, "run", forbidden)
    with pytest.raises(ValueError):
        app.run_pipeline(query, pdf, k, threshold)


def test_invalid_pdf_signature(pdf):
    pdf.write_bytes(b"not a PDF")
    with pytest.raises(ValueError, match="PDF"):
        app.run_pipeline("q", pdf)


@pytest.mark.parametrize("kind", ["timeout", "missing-python", "nonzero", "missing-result"])
def test_backend_failures_are_value_errors(pdf, monkeypatch, kind):
    def fail(*args, **kwargs):
        if kind == "timeout":
            raise subprocess.TimeoutExpired("backend", 1)
        if kind == "missing-python":
            raise FileNotFoundError("python")
        return SimpleNamespace(returncode=1 if kind == "nonzero" else 0, stderr="failure")

    monkeypatch.setattr(app.subprocess, "run", fail)
    with pytest.raises(ValueError):
        app.run_pipeline("q", pdf)


def test_default_export_directory_initializes_once_under_concurrency(result, monkeypatch):
    import time

    original = app.tempfile.TemporaryDirectory
    created = []

    def slow_directory(*args, **kwargs):
        # Widen the first-use race window without relying on probabilistic scheduling.
        time.sleep(0.02)
        directory = original(*args, **kwargs)
        created.append(directory)
        return directory

    monkeypatch.setattr(app, "_EXPORT_DIR", None)
    monkeypatch.setattr(app.tempfile, "TemporaryDirectory", slow_directory)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            exports = list(pool.map(lambda _: app.export_report(result), range(16)))
        assert len(created) == 1
        assert len({Path(paths[0]).parent for paths in exports}) == 16
        assert all(Path(path).is_file() for paths in exports for path in paths)
    finally:
        for directory in created:
            directory.cleanup()


def test_exported_html_anchor_targets_are_unique_and_highlightable(result, tmp_path):
    from html.parser import HTMLParser

    class Anchors(HTMLParser):
        def __init__(self):
            super().__init__()
            self.targets = []
            self.links = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.targets.append(attrs["id"])
            if tag == "a" and attrs.get("href", "").startswith("#"):
                self.links.append(attrs["href"][1:])

    evidence = copy.deepcopy(result["sentences"][0]["evidences"][0])
    result["sentences"][0]["evidences"].append(evidence)
    result["sentences"].append(copy.deepcopy(result["sentences"][0]))
    paths = app.export_report(result, tmp_path)
    content = Path(paths[3]).read_text("utf-8")
    anchors = Anchors()
    anchors.feed(content)
    assert len(anchors.targets) == len(set(anchors.targets)) == 4
    assert set(anchors.links) == set(anchors.targets)
    assert ".trace-evidence:target" in content
    assert "https://" not in content and "<script" not in content
