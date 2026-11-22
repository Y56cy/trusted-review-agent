"""P0 生成-校验-重写闭环的契约测试：全部用确定性替身，不碰真实模型。

覆盖改写闭环的分支：
  1. 无证据句改写后得到证据 → 用改写句替换原句、review 同步重建；
  2. 改写返回 [未能溯源] → 保留原句、空证据；
  3. 改写后仍无证据 → 保留原句、空证据；
  4. 全部有证据 → 不触发改写；
  5. 改写抛异常 → 退化为「未能溯源」，不拖垮整个结果。
"""

import importlib.util
import sys
import types

import app

PARAGRAPH = {"source": "论文", "para_id": 1, "page": 2, "text": "证据原文"}


def _evidence():
    return [{**PARAGRAPH, "nli_label": "entailment", "nli_prob": 0.9}]


def _make_pipeline(monkeypatch, *, generate, rewrite_sentence, match_sentence, split_sentences):
    """用替身模块加载 src/pipeline.py，返回 (module, seen)。"""
    seen = {"prompts": [], "rewrites": [], "matches": []}

    def parse_pdf(path):
        return [dict(PARAGRAPH)]

    class Retriever:
        def build_index(self, paragraphs):
            self.paragraphs = paragraphs

        def search(self, query, top_k=3):
            return list(self.paragraphs[:top_k])

    def gen(prompt):
        seen["prompts"].append(prompt)
        return generate()

    def rew(sentence, candidates):
        seen["rewrites"].append((sentence, list(candidates)))
        return rewrite_sentence(sentence, candidates)

    def mtch(sentence, candidates, threshold=0.7):
        seen["matches"].append((sentence, threshold))
        return match_sentence(sentence, candidates, threshold)

    mods = {
        "parser": {"parse_pdf": parse_pdf},
        "retriever": {"Retriever": Retriever},
        "generator": {"generate": gen, "rewrite_sentence": rew, "UNSUPPORTED": "[未能溯源]"},
        "match": {"split_sentences": split_sentences, "match_sentence": mtch},
    }
    for name, attrs in mods.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    spec = importlib.util.spec_from_file_location("pipeline_rewrite_test", app.SRC / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, seen


def test_rewrite_grounds_unsupported_sentence(monkeypatch):
    def match(sentence, candidates, threshold):
        return [] if sentence == "原句。" else _evidence()

    module, seen = _make_pipeline(
        monkeypatch,
        generate=lambda: "原句。有证据句。",
        rewrite_sentence=lambda s, c: "改写句。",
        match_sentence=match,
        split_sentences=lambda text: ["原句。", "有证据句。"],
    )

    result = module.run_review("问题", ["论文.pdf"], top_k=3, nli_threshold=0.7)

    assert [r[0] for r in seen["rewrites"]] == ["原句。"]
    assert [s["sentence"] for s in result["sentences"]] == ["改写句。", "有证据句。"]
    assert all(s["evidences"] for s in result["sentences"])
    assert result["review"] == "改写句。有证据句。"


def test_rewrite_returns_unsupported_marker_keeps_original(monkeypatch):
    def match(sentence, candidates, threshold):
        return [] if sentence == "原句。" else _evidence()

    module, seen = _make_pipeline(
        monkeypatch,
        generate=lambda: "原句。有证据句。",
        rewrite_sentence=lambda s, c: "[未能溯源]",
        match_sentence=match,
        split_sentences=lambda text: ["原句。", "有证据句。"],
    )

    result = module.run_review("问题", ["论文.pdf"], top_k=3, nli_threshold=0.7)

    assert [s["sentence"] for s in result["sentences"]] == ["原句。", "有证据句。"]
    assert result["sentences"][0]["evidences"] == []
    assert result["sentences"][1]["evidences"]
    assert result["review"] == "原句。有证据句。"  # 无回填，review 不变


def test_rewrite_still_unsupported_keeps_original(monkeypatch):
    def match(sentence, candidates, threshold):
        return []  # 原句与改写句都无证据

    module, seen = _make_pipeline(
        monkeypatch,
        generate=lambda: "原句。",
        rewrite_sentence=lambda s, c: "改写句。",
        match_sentence=match,
        split_sentences=lambda text: ["原句。"],
    )

    result = module.run_review("问题", ["论文.pdf"], top_k=3, nli_threshold=0.7)

    assert [r[0] for r in seen["rewrites"]] == ["原句。"]
    assert [s["sentence"] for s in result["sentences"]] == ["原句。"]
    assert result["sentences"][0]["evidences"] == []
    assert result["review"] == "原句。"


def test_all_grounded_skips_rewrite(monkeypatch):
    def match(sentence, candidates, threshold):
        return _evidence()

    module, seen = _make_pipeline(
        monkeypatch,
        generate=lambda: "有证据句。",
        rewrite_sentence=lambda s, c: "不应被调用。",
        match_sentence=match,
        split_sentences=lambda text: ["有证据句。"],
    )

    result = module.run_review("问题", ["论文.pdf"], top_k=3, nli_threshold=0.7)

    assert seen["rewrites"] == []
    assert result["review"] == "有证据句。"


def test_rewrite_exception_falls_back_to_unsupported(monkeypatch):
    def match(sentence, candidates, threshold):
        return [] if sentence == "原句。" else _evidence()

    def broken_rewrite(sentence, candidates):
        raise RuntimeError("llama-server 宕机")

    module, seen = _make_pipeline(
        monkeypatch,
        generate=lambda: "原句。有证据句。",
        rewrite_sentence=broken_rewrite,
        match_sentence=match,
        split_sentences=lambda text: ["原句。", "有证据句。"],
    )

    result = module.run_review("问题", ["论文.pdf"], top_k=3, nli_threshold=0.7)

    assert [s["sentence"] for s in result["sentences"]] == ["原句。", "有证据句。"]
    assert result["sentences"][0]["evidences"] == []
    assert result["sentences"][1]["evidences"]
