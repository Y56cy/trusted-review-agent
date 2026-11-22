"""P1 综述扩写智能体（占位符法）的契约测试：全部用确定性替身，不碰真实模型。"""

import importlib.util
import re
import sys
import types

import app


def _make_pipeline(monkeypatch, *, generate, split_sentences):
    mods = {
        "parser": {"parse_pdf": lambda path: []},
        "retriever": {"Retriever": type("Retriever", (), {})},
        "generator": {
            "generate": generate,
            "rewrite_sentence": lambda s, c: "[未能溯源]",
            "UNSUPPORTED": "[未能溯源]",
        },
        "match": {
            "match_sentence": lambda s, c, t=0.7: [],
            "split_sentences": split_sentences,
        },
    }
    for name, attrs in mods.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("pipeline_expand_test", app.SRC / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split(text):
    text = text.strip()
    if not text:
        return []
    return [p for p in re.split(r"(?<=[。！？])", text) if p.strip()]


def _result():
    return {
        "query": "问题",
        "review": "关键句一。关键句二。",
        "sentences": [
            {
                "sentence": "关键句一。",
                "evidences": [
                    {
                        "source": "论文",
                        "para_id": 1,
                        "page": 2,
                        "text": "证据一",
                        "nli_label": "entailment",
                        "nli_prob": 0.9,
                    }
                ],
            },
            {
                "sentence": "关键句二。",
                "evidences": [
                    {
                        "source": "论文",
                        "para_id": 2,
                        "page": 3,
                        "text": "证据二",
                        "nli_label": "entailment",
                        "nli_prob": 0.8,
                    }
                ],
            },
        ],
    }


def test_expand_review_inserts_key_sentences_and_marks_structural(monkeypatch):
    scaffold = "这是引言。<K1>这是过渡。<K2>这是总结。"
    module = _make_pipeline(monkeypatch, generate=lambda prompt: scaffold, split_sentences=_split)

    expanded = module.expand_review(_result())

    assert [s["sentence"] for s in expanded["sentences"]] == [
        "这是引言。",
        "关键句一。",
        "这是过渡。",
        "关键句二。",
        "这是总结。",
    ]
    assert [s["structural"] for s in expanded["sentences"]] == [True, False, True, False, True]
    assert expanded["sentences"][1]["evidences"][0]["para_id"] == 1
    assert expanded["sentences"][3]["evidences"][0]["para_id"] == 2
    assert expanded["review"] == "这是引言。关键句一。这是过渡。关键句二。这是总结。"


def test_expand_review_no_key_sentences_returns_original(monkeypatch):
    original = {
        "query": "问题",
        "review": "无证据句。",
        "sentences": [{"sentence": "无证据句。", "evidences": []}],
    }
    module = _make_pipeline(
        monkeypatch, generate=lambda prompt: "不应被调用", split_sentences=_split
    )

    assert module.expand_review(original) is original


def test_expand_review_missing_placeholder_appends_key(monkeypatch):
    scaffold = "这是引言。<K1>这是总结。"  # 漏了 <K2>
    module = _make_pipeline(monkeypatch, generate=lambda prompt: scaffold, split_sentences=_split)

    expanded = module.expand_review(_result())

    assert [s["sentence"] for s in expanded["sentences"]] == [
        "这是引言。",
        "关键句一。",
        "这是总结。",
        "关键句二。",
    ]
    assert expanded["sentences"][-1]["sentence"] == "关键句二。"
    assert expanded["sentences"][-1]["structural"] is False
    assert expanded["sentences"][-1]["evidences"][0]["para_id"] == 2


def test_annotate_citations_adds_markers_and_references(monkeypatch):
    module = _make_pipeline(monkeypatch, generate=lambda prompt: "", split_sentences=_split)

    result = {
        "query": "问题",
        "review": "句一。句二。句三。",
        "sentences": [
            {
                "sentence": "句一。",
                "evidences": [{"source": "论文A", "para_id": 1, "page": 2, "text": "证据甲"}],
            },
            {
                "sentence": "句二。",
                "evidences": [
                    {"source": "论文A", "para_id": 1, "page": 2, "text": "证据甲"},
                    {"source": "论文B", "para_id": 3, "page": 5, "text": "证据乙"},
                ],
            },
            {"sentence": "句三。", "evidences": []},
        ],
    }

    annotated = module.annotate_citations(result)

    assert annotated["review"] == "句一[1]。句二[1][2]。句三。"
    assert annotated["references"] == [
        "[1]《论文A》第2页 第1段：证据甲",
        "[2]《论文B》第5页 第3段：证据乙",
    ]
