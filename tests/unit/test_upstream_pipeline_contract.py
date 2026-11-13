"""Exercise the current upstream orchestrator with explicit deterministic substitutes.

These verify contracts, never real PDFs, model quality or formal evaluation.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

import app


@pytest.fixture
def pipeline(monkeypatch):
    seen = {"paths": [], "prompts": [], "matches": []}

    def parse_pdf(path):
        seen["paths"].append(path)
        return [{"source": Path(path).stem, "para_id": 1, "page": 2, "text": "合成证据"}]

    class Retriever:
        def build_index(self, paragraphs):
            self.paragraphs = paragraphs

        def search(self, query, top_k):
            return self.paragraphs[:top_k]

    def generate(prompt):
        seen["prompts"].append(prompt)
        return "合成生成句。"

    def match_sentence(sentence, candidates, threshold):
        seen["matches"].append((sentence, threshold))
        return [{**p, "nli_label": "entailment", "nli_prob": 0.9} for p in candidates]

    modules = {
        "parser": {"parse_pdf": parse_pdf},
        "retriever": {"Retriever": Retriever},
        "generator": {"generate": generate},
        "match": {"split_sentences": lambda text: [text], "match_sentence": match_sentence},
    }
    for name, functions in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(functions)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("pipeline_contract_test", app.SRC / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, seen


def test_current_multi_pdf_pipeline_survives_ui_export_roundtrip(pipeline, tmp_path):
    module, seen = pipeline
    paths = [f"paper-{n}.pdf" for n in range(5)]
    result = module.run_review("合成测试问题", paths, top_k=5, nli_threshold=0.6)
    assert seen["paths"] == paths
    assert seen["matches"] == [("合成生成句。", 0.6)]
    assert len(seen["prompts"]) == 1
    for n in range(5):
        assert f"paper-{n} 第2页 第1段" in seen["prompts"][0]
    valid = app.validate_result(result)
    assert len(valid["sentences"][0]["evidences"]) == 5
    paths = app.export_report(valid, tmp_path, source_name="合成契约测试", is_demo=True)
    restored, metadata = app.load_result(paths[2])
    assert restored == valid
    assert metadata["is_demo"] is True
    rendered = app.render_workspace(restored, anchor_prefix="contract-")
    assert all(f"paper-{n}" in rendered for n in range(5))
    assert "contract-ev-1-5" in rendered


@pytest.mark.parametrize("path", ["one.pdf", Path("one.pdf")])
def test_current_pipeline_single_path_remains_compatible(pipeline, path):
    module, seen = pipeline
    result = module.run_review("合成单篇问题", path)
    assert seen["paths"] == [path]
    assert app.validate_result(result)["sentences"][0]["evidences"][0]["source"] == "one"
