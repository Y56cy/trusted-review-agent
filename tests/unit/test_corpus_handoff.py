"""Synthetic contract checks for member A's ce04a5e corpus export."""

import copy
import json

import pytest

import app


@pytest.fixture
def corpus():
    paragraphs = [
        {k: e[k] for k in ("source", "page", "para_id", "text")}
        for e in app.DEMO_RESULT["sentences"][0]["evidences"]
    ]
    digest = app._paragraph_checksum(paragraphs)
    return {
        "dataset_type": "synthetic",
        "paragraphs": paragraphs,
        "parsed_paragraphs_sha256": digest,
        "manifest": {
            "parsed_paragraphs_sha256": digest,
            "paragraph_count": len(paragraphs),
            "documents": [
                {
                    "source": p["source"],
                    "paragraph_count": 1,
                    "parsed_paragraphs_sha256": app._paragraph_checksum([p]),
                }
                for p in paragraphs
            ],
        },
    }


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_valid_snapshot_with_colliding_paragraph_ids(corpus):
    index, checksum = app.validate_frozen_corpus(corpus)
    assert len(index) == 2
    assert checksum == corpus["parsed_paragraphs_sha256"]


@pytest.mark.parametrize("bad", [None, [], {}, {"paragraphs": []}])
def test_invalid_snapshots(bad):
    with pytest.raises(ValueError):
        app.validate_frozen_corpus(bad)


@pytest.mark.parametrize(
    "field,value", [("source", ""), ("source", 1), ("text", ""), ("page", True), ("para_id", 0)]
)
def test_invalid_paragraphs(corpus, field, value):
    corpus["paragraphs"][0][field] = value
    with pytest.raises(ValueError):
        app.validate_frozen_corpus(corpus)


def test_duplicate_identity_rejected(corpus):
    corpus["paragraphs"].append(copy.deepcopy(corpus["paragraphs"][0]))
    with pytest.raises(ValueError, match="重复"):
        app.validate_frozen_corpus(corpus)


@pytest.mark.parametrize(
    "field,value",
    [
        ("paragraph_count", 3),
        ("paragraph_count", True),
        ("parsed_paragraphs_sha256", "bad"),
        ("documents", []),
    ],
)
def test_manifest_tampering(corpus, field, value):
    corpus["manifest"][field] = value
    with pytest.raises(ValueError):
        app.validate_frozen_corpus(corpus)


def test_document_tampering(corpus):
    corpus["manifest"]["documents"][0]["parsed_paragraphs_sha256"] = "bad"
    with pytest.raises(ValueError):
        app.validate_frozen_corpus(corpus)


def test_snapshot_tampering(corpus):
    corpus["paragraphs"][0]["text"] = "changed"
    with pytest.raises(ValueError, match="校验值"):
        app.validate_frozen_corpus(corpus)


def test_check_good_empty_and_bad_results(tmp_path, corpus):
    c = write(tmp_path / "corpus.json", corpus)
    r = write(tmp_path / "result.json", app.DEMO_RESULT)
    text = app.check_corpus_evidence(c, r)
    assert "证据字段一致" in text and "一致 2 条" in text and "无证据句 1" in text
    assert "不代表" in text
    assert "synthetic" in text and app.DEMO_NOTICE in text
    result = copy.deepcopy(app.DEMO_RESULT)
    result["sentences"][0]["evidences"] = []
    write(r, result)
    assert "无证据可核对" in app.check_corpus_evidence(c, r)
    result = copy.deepcopy(app.DEMO_RESULT)
    result["sentences"][0]["evidences"][0]["page"] = 99
    result["sentences"][0]["evidences"][1]["text"] = "changed"
    write(r, result)
    text = app.check_corpus_evidence(c, r)
    assert "不一致 2 条" in text and "page" in text and "text" in text


@pytest.mark.parametrize("source,expected", [("", "缺少文献名"), ("unknown", "不在冻结语料")])
def test_unknown_source_not_inferred(tmp_path, corpus, source, expected):
    result = copy.deepcopy(app.DEMO_RESULT)
    result["sentences"][0]["evidences"][0]["source"] = source
    text = app.check_corpus_evidence(
        write(tmp_path / "c.json", corpus), write(tmp_path / "r.json", result)
    )
    assert expected in text


def test_envelope_demo_is_retained(tmp_path, corpus):
    files = app.export_report(app.DEMO_RESULT, source_name="synthetic", is_demo=True)
    text = app.check_corpus_evidence(write(tmp_path / "c.json", corpus), files[2])
    assert app.DEMO_NOTICE in text


def test_bad_input_clears_previous_check(tmp_path, corpus):
    c = write(tmp_path / "c.json", corpus)
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    assert "未完成核对" in app.check_corpus_evidence(c, bad)
    assert "未完成核对" in app.check_corpus_evidence(None, None)
    bad.write_bytes(b" " * (app.MAX_JSON_BYTES + 1))
    assert "5 MiB" in app.check_corpus_evidence(c, bad)
