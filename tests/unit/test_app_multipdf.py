"""Independent multi-document request and provenance regression contracts."""

import copy
import csv
import html
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import app


@pytest.fixture
def multi_result():
    return {
        "query": "跨文献问题",
        "review": "两份文献共同提供依据。",
        "sentences": [
            {
                "sentence": "两份文献共同提供依据。",
                "evidences": [
                    {
                        "source": source,
                        "page": 1,
                        "para_id": 1,
                        "text": text,
                        "nli_label": "entailment",
                        "nli_prob": 0.8,
                    }
                    for source, text in [("论文乙", "排名第一的证据"), ("论文甲", "排名第二的证据")]
                ],
            }
        ],
    }


def make_pdf(folder, name, size=None):
    path = folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(b"%PDF-1.7\n")
        if size is not None:
            stream.truncate(size)
    return path


@pytest.mark.parametrize("kind", ["str", "path", "list", "tuple"])
def test_request_normalizes_single_and_multiple_paths(tmp_path, kind):
    first = make_pdf(tmp_path, "first.pdf")
    second = make_pdf(tmp_path, "second.PDF")
    value = {
        "str": str(first),
        "path": first,
        "list": [str(first), second],
        "tuple": (first, str(second)),
    }[kind]
    query, paths = app._validate_request(" 问题 ", value, 3, 0.7)
    assert query == "问题"
    assert paths == ([first, second] if kind in ("list", "tuple") else [first])
    assert all(isinstance(path, Path) for path in paths)


@pytest.mark.parametrize("value", [None, [], (), "", [None], [23], {}])
def test_invalid_path_containers_rejected(value):
    with pytest.raises(ValueError):
        app._validate_request("q", value, 3, 0.7)


def test_every_uploaded_pdf_is_validated(tmp_path, monkeypatch):
    first = make_pdf(tmp_path, "good.pdf")
    second = make_pdf(tmp_path, "bad.pdf")
    second.write_bytes(b"this is not a PDF")
    monkeypatch.setattr(app.subprocess, "run", lambda *a, **k: pytest.fail("backend started"))
    with pytest.raises(ValueError):
        app.run_pipeline("q", [first, second])


def test_pdf_count_limit_and_boundary(tmp_path):
    paths = [make_pdf(tmp_path, f"paper-{i}.pdf") for i in range(11)]
    assert len(app._validate_request("q", paths[:10], 3, 0.7)[1]) == 10
    with pytest.raises(ValueError):
        app._validate_request("q", paths, 3, 0.7)


def test_pdf_individual_and_aggregate_size_limits(tmp_path):
    mib = 1024 * 1024
    paths = [make_pdf(tmp_path, f"paper-{i}.pdf", 50 * mib) for i in range(4)]
    assert app._validate_request("q", paths, 3, 0.7)[1] == paths
    with pytest.raises(ValueError):
        app._validate_request("q", paths + [make_pdf(tmp_path, "extra.pdf")], 3, 0.7)
    oversized = make_pdf(tmp_path, "oversized.pdf", 50 * mib + 1)
    with pytest.raises(ValueError):
        app._validate_request("q", [oversized], 3, 0.7)


@pytest.mark.parametrize("names", [("Paper.pdf", "paper.PDF"), ("é.pdf", "e\u0301.pdf")])
def test_duplicate_source_stems_rejected_across_folders(tmp_path, names):
    paths = [make_pdf(tmp_path / str(i), name) for i, name in enumerate(names)]
    with pytest.raises(ValueError):
        app._validate_request("q", paths, 3, 0.7)


def test_pipeline_request_uses_pdf_paths_and_preserves_sources(tmp_path, monkeypatch, multi_result):
    paths = [make_pdf(tmp_path, name) for name in ("one.pdf", "two.pdf")]

    def run(command, **kwargs):
        request = json.loads(Path(command[-2]).read_text("utf-8"))
        assert request == {
            "query": "q",
            "pdf_paths": list(map(str, paths)),
            "top_k": 3,
            "nli_threshold": 0.7,
        }
        Path(command[-1]).write_text(json.dumps(multi_result), "utf-8")
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(app.subprocess, "run", run)
    assert app.run_pipeline("q", paths) == multi_result


def test_worker_passes_multi_pdf_contract_to_backend(tmp_path, monkeypatch, multi_result):
    request = {"query": "q", "pdf_paths": ["one.pdf", "two.pdf"], "top_k": 3, "nli_threshold": 0.7}
    request_file, result_file = tmp_path / "request.json", tmp_path / "result.json"
    request_file.write_text(json.dumps(request), "utf-8")

    def run_review(**kwargs):
        assert kwargs == request
        return multi_result

    monkeypatch.setattr(app, "environment_issues", lambda: [])
    monkeypatch.setitem(sys.modules, "pipeline", SimpleNamespace(run_review=run_review))
    assert app._pipeline_worker(str(request_file), str(result_file)) == 0
    assert json.loads(result_file.read_text("utf-8")) == multi_result


@pytest.mark.parametrize("source", [None, 1, True, [], {}])
def test_non_string_evidence_source_rejected(multi_result, source):
    multi_result["sentences"][0]["evidences"][0]["source"] = source
    with pytest.raises(ValueError):
        app.validate_result(multi_result)


def test_source_preserved_verbatim_without_normalization(multi_result):
    source = "  E\u0301tude / 文献.PDF  "
    multi_result["sentences"][0]["evidences"][0]["source"] = source
    assert app.validate_result(multi_result) == multi_result


@pytest.mark.parametrize("missing", [True, False])
def test_legacy_source_never_inferred_from_report_label(tmp_path, multi_result, missing):
    for evidence in multi_result["sentences"][0]["evidences"]:
        if missing:
            evidence.pop("source")
        else:
            evidence["source"] = ""
    envelope = {
        "schema_version": 1,
        "metadata": {"source_name": "claimed-paper.pdf", "is_demo": False},
        "result": multi_result,
    }
    imported = tmp_path / "legacy.json"
    imported.write_text(json.dumps(envelope), "utf-8")
    checked, _ = app.load_result(imported)
    assert [e["source"] for e in checked["sentences"][0]["evidences"]] == ["", ""]
    assert "文献来源未知" in app.render_provenance(checked)
    exported = app.export_report(checked, tmp_path, source_name="claimed-paper.pdf")
    for index in (0, 3):
        assert "文献来源未知" in Path(exported[index]).read_text("utf-8")
    with Path(exported[1]).open(encoding="utf-8-sig", newline="") as stream:
        assert all(row["文献名"] == "文献来源未知" for row in csv.DictReader(stream))


def test_same_paragraph_id_keeps_distinct_sources_and_rank(tmp_path, multi_result):
    assert app.validate_result(multi_result) == multi_result
    rendered = app.render_provenance(multi_result)
    for css_class in ("trace-columns", "trace-review", "trace-sources"):
        assert css_class in rendered
    assert 'id="ev-1-1"' in rendered and 'id="ev-1-2"' in rendered
    exported = app.export_report(multi_result, tmp_path, source_name="报告标签")
    for content in [rendered, *(Path(exported[i]).read_text("utf-8") for i in (0, 3))]:
        assert content.index("排名第一的证据") < content.index("排名第二的证据")
        assert "论文乙" in content and "论文甲" in content
    with Path(exported[1]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["文献名"] for row in rows] == ["论文乙", "论文甲"]
    assert [row["段号"] for row in rows] == ["1", "1"]
    assert [row["来源"] for row in rows] == ["报告标签", "报告标签"]
    assert app.load_result(exported[2])[0] == multi_result


@pytest.mark.parametrize(
    "payload",
    [
        '<script>alert("x")</script>',
        "[link](https://invalid)|x",
        '=HYPERLINK("https://invalid")',
        "  @SUM(A1)",
    ],
)
def test_source_export_injection_safety(tmp_path, multi_result, payload):
    multi_result["sentences"][0]["evidences"][0]["source"] = payload
    original = copy.deepcopy(multi_result)
    exported = app.export_report(multi_result, tmp_path)
    rendered = app.render_provenance(multi_result)
    for content in (rendered, Path(exported[3]).read_text("utf-8")):
        assert html.escape(payload) in content
        assert "<script>" not in content
    md = Path(exported[0]).read_text("utf-8")
    assert app._markdown(payload) in md
    if payload.startswith("[link]"):
        assert payload not in md
    with Path(exported[1]).open(encoding="utf-8-sig", newline="") as stream:
        source = next(csv.DictReader(stream))["文献名"]
    assert source == ("'" + payload if payload.lstrip().startswith(("=", "@")) else payload)
    assert json.loads(Path(exported[2]).read_text("utf-8"))["result"] == original
    assert multi_result == original


def test_demo_demonstrates_multiple_named_sources():
    sources = {
        e["source"]
        for item in app.validate_result(app.DEMO_RESULT)["sentences"]
        for e in item["evidences"]
        if e["source"]
    }
    assert len(sources) >= 2


def test_sentence_navigation_includes_empty_groups_and_reset(tmp_path, multi_result):
    from html.parser import HTMLParser

    class Navigation(HTMLParser):
        def __init__(self):
            super().__init__()
            self.ids = []
            self.links = []
            self.groups = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if "id" in attrs:
                self.ids.append(attrs["id"])
            if tag == "a" and attrs.get("href", "").startswith("#"):
                self.links.append(attrs["href"][1:])
            if "trace-evidence-group" in attrs.get("class", "").split():
                self.groups.append(attrs["id"])

    multi_result["sentences"].append({"sentence": "暂无证据的句子", "evidences": []})
    paths = app.export_report(multi_result, tmp_path)
    for content in (app.render_provenance(multi_result), Path(paths[3]).read_text("utf-8")):
        navigation = Navigation()
        navigation.feed(content)
        assert len(navigation.ids) == len(set(navigation.ids))
        assert set(navigation.links) <= set(navigation.ids)
        assert navigation.groups == ["sentence-1", "sentence-2"]
        assert {"sentence-1", "sentence-2", "trace-all", "ev-1-1", "ev-1-2"} <= set(
            navigation.links
        )
        assert "查看全部证据" in content
        assert "未匹配到证据" in content
        assert "<script" not in content


def test_upstream_pipeline_five_pdf_contract_without_models(tmp_path, monkeypatch):
    """Execute the real orchestration module with only model/I/O boundaries stubbed."""
    paths = [make_pdf(tmp_path, f"论文{i}.pdf") for i in range(5)]
    expected_sources = [path.stem for path in paths]
    parsed_paths, indexed_sources, matched_sources, prompts = [], [], [], []
    review = "五份文献共同支持本句。"

    def parse_pdf(path):
        parsed_paths.append(Path(path))
        return [
            {
                "source": Path(path).stem,
                "page": 1,
                "para_id": 1,
                "text": f"来自{Path(path).stem}的证据",
            }
        ]

    class Retriever:
        def build_index(self, paragraphs):
            self.paragraphs = copy.deepcopy(paragraphs)
            indexed_sources.extend(item["source"] for item in paragraphs)

        def search(self, query, top_k=3):
            return copy.deepcopy(self.paragraphs[:top_k])

    def generate(prompt):
        prompts.append(prompt)
        return review

    def match_sentence(sentence, candidates, threshold=0.7):
        assert sentence == review
        assert threshold == 0.8
        matched_sources.extend(item["source"] for item in candidates)
        return [{**item, "nli_label": "entailment", "nli_prob": 0.9} for item in candidates]

    stubs = {
        "parser": SimpleNamespace(parse_pdf=parse_pdf),
        "retriever": SimpleNamespace(Retriever=Retriever),
        "generator": SimpleNamespace(generate=generate),
        "match": SimpleNamespace(
            split_sentences=lambda text: [text], match_sentence=match_sentence
        ),
    }
    # monkeypatch restores every module entry, including pre-existing modules.
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "upstream_pipeline_contract", app.SRC / "pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
    result = pipeline.run_review("综合五篇文献", pdf_paths=paths, top_k=5, nli_threshold=0.8)
    assert parsed_paths == paths
    assert indexed_sources == matched_sources == expected_sources
    assert prompts and all(source in prompts[0] for source in expected_sources)
    checked = app.validate_result(result)
    assert checked["query"] == "综合五篇文献"
    assert checked["review"] == review
    assert len(checked["sentences"]) == 1
    assert [item["source"] for item in checked["sentences"][0]["evidences"]] == expected_sources
