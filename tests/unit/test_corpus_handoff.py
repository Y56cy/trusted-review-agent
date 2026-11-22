"""Synthetic contract checks for member A's ce04a5e corpus export."""

import copy
import csv
import json
from pathlib import Path

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


def test_check_good_empty_and_bad_results(corpus):
    index, _ = app.validate_frozen_corpus(corpus)
    result = copy.deepcopy(app.DEMO_RESULT)
    checked = app.check_corpus_evidence(result, index)
    assert checked["sentences"][1]["evidences"] == []
    assert [e["corpus_verified"] for e in checked["sentences"][0]["evidences"]] == [True, True]
    assert result == app.DEMO_RESULT
    evidence = result["sentences"][0]["evidences"]
    evidence[0]["page"] = 99
    evidence[1]["text"] = "changed"
    checked = app.check_corpus_evidence(result, index)
    assert [e["corpus_verified"] for e in checked["sentences"][0]["evidences"]] == [False, False]


@pytest.mark.parametrize("source", ["", "unknown"])
def test_unknown_source_not_inferred(corpus, source):
    index, _ = app.validate_frozen_corpus(corpus)
    result = copy.deepcopy(app.DEMO_RESULT)
    result["sentences"][0]["evidences"][0]["source"] = source
    checked = app.check_corpus_evidence(result, index)
    assert checked["sentences"][0]["evidences"][0]["corpus_verified"] is False


@pytest.mark.parametrize("value", [0, 1, "false", {}])
def test_corpus_status_does_not_coerce_invalid_values(value):
    result = copy.deepcopy(app.DEMO_RESULT)
    result["sentences"][0]["evidences"][0]["corpus_verified"] = value
    with pytest.raises(ValueError, match="corpus_verified"):
        app.validate_result(result)


def test_load_frozen_corpus_replaces_or_clears_previous_state(tmp_path, corpus):
    valid = write(tmp_path / "corpus.json", corpus)
    index, status = app.load_frozen_corpus(valid)
    assert len(index) == 2 and "已加载冻结语料" in status
    assert app.load_frozen_corpus(None) == (None, "未加载冻结语料。")

    corpus["paragraphs"][0]["text"] = "tampered"
    assert app.load_frozen_corpus(write(valid, corpus))[0] is None
    bad = tmp_path / "bad.json"
    bad.write_text("{", encoding="utf-8")
    state, status = app.load_frozen_corpus(bad)
    assert state is None and "未加载" in status
    bad.write_bytes(b" " * (app.MAX_JSON_BYTES + 1))
    state, status = app.load_frozen_corpus(bad)
    assert state is None and "5 MiB" in status


def test_verification_status_is_preserved_in_all_reports(tmp_path, corpus):
    index, _ = app.validate_frozen_corpus(corpus)
    result = copy.deepcopy(app.DEMO_RESULT)
    evidence = result["sentences"][0]["evidences"]
    evidence[1]["page"] = 99
    annotated = app.check_corpus_evidence(result, index)
    unchecked = app.check_corpus_evidence(result, None)["sentences"][0]["evidences"][0]
    annotated["sentences"][0]["evidences"].append(unchecked)
    md, csv_path, json_path, html_path = map(Path, app.export_report(annotated, tmp_path))
    assert app.load_result(json_path)[0] == annotated
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row, value in zip(rows[:-1], [True, False, None], strict=True):
        text = app.CORPUS_STATUS[value][1]
        assert row["语料一致性校验"] == text
        assert text in md.read_text("utf-8")
        assert text in html_path.read_text("utf-8")
    assert rows[-1]["语料一致性校验"] == ""
