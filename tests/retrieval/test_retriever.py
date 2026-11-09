"""确定性模型替身只验证排序逻辑，不能用于效果结论。"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import retriever as module


class FakeEncoder:
    def __init__(self, mapping=None):
        self.mapping = mapping or {}

    def encode(self, texts, **kwargs):
        return np.array([self.mapping.get(text, [1.0, 0.0]) for text in texts])


def paragraph(i, text):
    return {"para_id": i, "page": 1, "text": text}


class RetrievalTests(unittest.TestCase):
    def make(self, corpus=None, mapping=None):
        with patch.object(module, "_load_embedding_model", return_value=FakeEncoder(mapping)):
            result = module.Retriever()
        if corpus is not None:
            result.build_index(corpus)
        return result

    def test_tokenization(self):
        tokens = module.tokenize("BGE，中文 A7 3.14 Qwen2.5-7B！")
        self.assertIn("bge", tokens)
        self.assertTrue(any("中" in word for word in tokens))
        self.assertTrue(any(any(c.isdigit() for c in word) for word in tokens))
        self.assertTrue(all(word.isalnum() for word in tokens))

    def test_keyword_exact_match_and_contract(self):
        corpus = [
            paragraph(1, "BM25 检索采用词频和文档频率。"),
            paragraph(2, "天气预报记录气温。"),
            paragraph(3, "图书馆办理读者登记。"),
        ]
        r = self.make(corpus)
        hits = r.search("BM25", method="bm25")
        self.assertEqual([h["para_id"] for h in hits], [1])
        self.assertEqual(set(hits[0]), {"text", "page", "para_id", "source", "score"})
        self.assertAlmostEqual(hits[0]["score"], 1)
        self.assertEqual(r.search("BM25"), r.search("BM25", method="hybrid"))

    def test_small_corpus_exact_keyword_keeps_positive_score(self):
        for corpus in (
            [paragraph(1, "BM25 检索科研材料")],
            [paragraph(1, "BM25 检索科研材料"), paragraph(2, "天气气温变化")],
        ):
            with self.subTest(size=len(corpus)):
                r = self.make(corpus)
                self.assertGreater(r.bm25.idf["bm25"], 0)
                self.assertEqual([h["para_id"] for h in r.search("BM25", method="bm25")], [1])

    def test_common_query_terms_are_not_discarded(self):
        r = self.make([paragraph(i, "中文科研材料") for i in range(1, 4)])
        self.assertEqual([h["para_id"] for h in r.search("科研", method="bm25")], [1, 2, 3])
        self.assertEqual(r.search("absentxyz", method="bm25"), [])

    def test_bm25_tiebreak_is_independent_of_vector_ranking(self):
        corpus = [paragraph(2, "中文乙"), paragraph(1, "中文甲")]
        r = self.make(corpus, {"中文甲": [0.2, 1.0], "中文乙": [1.0, 0.0]})
        with patch.object(r.bm25, "get_scores", return_value=np.array([1, 1])):
            self.assertEqual([h["para_id"] for h in r.search("查询", method="bm25")], [1, 2])

    def test_rrf_changes_order_keeps_vector_scores_and_deduplicates(self):
        corpus = [paragraph(i, f"中文段落内容{i}") for i in range(1, 4)]
        mapping = {
            corpus[0]["text"]: [0.2, 0.98],
            corpus[1]["text"]: [0.6, 0.8],
            corpus[2]["text"]: [1, 0],
        }
        r = self.make(corpus, mapping)
        # vector: 3,2,1; lexical: 2,1。RRF 应将双路均命中的 2 提到最前。
        with patch.object(r.bm25, "get_scores", return_value=np.array([1, 2, 0])):
            results = r.search("查询", top_k=3)
        self.assertEqual([h["para_id"] for h in results], [2, 1, 3])
        self.assertAlmostEqual(results[0]["score"], 0.6)
        self.assertLess(results[0]["score"], results[2]["score"])
        self.assertEqual(len({h["para_id"] for h in results}), 3)

    def test_rrf_tie_uses_vector_similarity(self):
        corpus = [paragraph(1, "中文甲"), paragraph(2, "中文乙")]
        r = self.make(corpus, {"中文乙": [0.6, 0.8]})
        with patch.object(r.bm25, "get_scores", return_value=np.array([1, 2])):
            self.assertEqual([h["para_id"] for h in r.search("查询")], [1, 2])

    def test_stable_id_tiebreak(self):
        r = self.make([paragraph(3, "中文丙"), paragraph(1, "中文甲"), paragraph(2, "中文乙")])
        for mode in ("vector", "hybrid"):
            self.assertEqual([h["para_id"] for h in r.search("xyz", method=mode)], [1, 2, 3])

    def test_no_keyword_fallback(self):
        r = self.make([paragraph(1, "中文科研"), paragraph(2, "天气气温")])
        self.assertEqual(r.search("zzzzzz"), r.search("zzzzzz", method="vector"))
        self.assertEqual(r.search("zzzzzz", method="bm25"), [])

    def test_language_filter_precedes_candidate_limit(self):
        corpus = [paragraph(i, f"English reference text {i}") for i in range(1, 32)]
        corpus += [paragraph(32, "中文证据科研材料。")]
        r = self.make(corpus, {corpus[-1]["text"]: [0.1, 1]})
        for method in module.METHODS:
            with self.subTest(method=method):
                self.assertEqual(
                    [h["para_id"] for h in r.search("中文证据", top_k=1, method=method)], [32]
                )
        self.assertEqual(r.search("查询", min_chinese_ratio=1), [])

    def test_candidate_limit_and_large_top_k(self):
        corpus = [paragraph(i, f"中文内容{i}") for i in range(1, 31)]
        r = self.make(corpus)
        scores = np.arange(1, 31)
        with patch.object(r.bm25, "get_scores", return_value=scores):
            self.assertEqual(len(r.search("问题", top_k=25)), 25)
            self.assertEqual(len(r.search("问题", top_k=100)), 30)

    def test_rrf_excludes_candidates_outside_both_top_twenty(self):
        corpus = [paragraph(i, f"中文材料{i}") for i in range(1, 46)]
        mapping = {p["text"]: [46 - p["para_id"], 1.0] for p in corpus}
        r = self.make(corpus, mapping)
        # 两路分别为 1..20、45..26；21..25 不在任何候选通道内。
        with patch.object(r.bm25, "get_scores", return_value=np.arange(1, 46)):
            ids = [h["para_id"] for h in r.search("查询", top_k=3)]
        self.assertEqual(ids, [1, 45, 2])

    def test_query_and_parameter_errors(self):
        r = self.make([paragraph(1, "中文材料")])
        self.assertEqual(r.search(" \n"), [])
        for params in (
            {"top_k": 0},
            {"top_k": -1},
            {"top_k": True},
            {"top_k": 1.5},
            {"method": "other"},
            {"min_chinese_ratio": float("nan")},
            {"min_chinese_ratio": 2},
            {"min_chinese_ratio": True},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                r.search("问题", **params)
        with self.assertRaises(ValueError):
            r.search(None)
        with self.assertRaises(RuntimeError):
            self.make().search("问题")

    def test_index_validation_and_replacement(self):
        r = self.make([paragraph(1, "旧中文材料")])
        for bad in (
            [],
            [paragraph(1, "")],
            [paragraph(1, "!!!")],
            [paragraph(1, "甲"), paragraph(1, "乙")],
            [{"text": "材料", "page": True, "para_id": 1}],
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                r.build_index(bad)
        # 失败不留下半更新的双索引。
        self.assertEqual(r.search("问题")[0]["para_id"], 1)
        r.build_index([paragraph(2, "新的科研材料")])
        self.assertEqual([h["para_id"] for h in r.search("科研")], [2])
        self.assertEqual(len(r.tokens), 1)

    def test_multi_source_paragraphs_are_distinct(self):
        # 多篇文献：同 para_id 但不同 source 不算重复
        corpus = [
            {"para_id": 1, "page": 1, "text": "甲文档段落内容", "source": "doc_a"},
            {"para_id": 1, "page": 1, "text": "乙文档段落内容", "source": "doc_b"},
        ]
        r = self.make(corpus)
        self.assertEqual(len(r.corpus), 2)
        self.assertEqual({h["source"] for h in r.search("段落")}, {"doc_a", "doc_b"})
        # 同一 source 下 para_id 重复仍报错
        with self.assertRaises(ValueError):
            self.make([
                {"para_id": 1, "page": 1, "text": "甲", "source": "doc_a"},
                {"para_id": 1, "page": 1, "text": "乙", "source": "doc_a"},
            ])

    def test_input_snapshot(self):
        corpus = [paragraph(1, "原始中文材料")]
        r = self.make(corpus)
        corpus[0]["text"] = "更改"
        self.assertEqual(r.corpus[0]["text"], "原始中文材料")

    def test_missing_model_help(self):
        with self.assertRaisesRegex(FileNotFoundError, "ModelScope"):
            module._load_embedding_model(ROOT / "models/does-not-exist")

    def test_default_path_ignores_cwd_and_explicit_path_wins(self):
        with patch.object(module, "_load_embedding_model", return_value=FakeEncoder()) as load:
            module.Retriever()
            self.assertEqual(load.call_args.args[0], ROOT / "models/bge-small-zh-v1.5")
            module.Retriever(ROOT / "custom-model")
            self.assertEqual(load.call_args.args[0], ROOT / "custom-model")

    def test_pipeline_consumes_search_results_without_contract_change(self):
        import importlib.util
        import types

        # 避免导入 NLI 权重；运行原 pipeline 的实际编排代码。
        corpus = [paragraph(1, "BM25 根据词频检索文献段落。")]
        parser_stub = types.ModuleType("parser")
        parser_stub.parse_pdf = lambda path: corpus
        generator_stub = types.ModuleType("generator")
        generator_stub.generate = lambda prompt: "BM25 根据词频检索文献。"
        match_stub = types.ModuleType("match")
        match_stub.split_sentences = lambda text: [text]
        match_stub.match_sentence = lambda sentence, candidates, threshold: [
            {k: c[k] for k in ("text", "page", "para_id")}
            | {"nli_label": "entailment", "nli_prob": 0.9}
            for c in candidates
        ]
        spec = importlib.util.spec_from_file_location("pipeline_contract", ROOT / "src/pipeline.py")
        pipeline = importlib.util.module_from_spec(spec)
        with (
            patch.dict(
                sys.modules,
                {"parser": parser_stub, "generator": generator_stub, "match": match_stub},
            ),
            patch.object(module, "_load_embedding_model", return_value=FakeEncoder()),
        ):
            spec.loader.exec_module(pipeline)
            result = pipeline.run_review("BM25", "not-a-real-pdf")
        self.assertEqual(set(result), {"query", "review", "sentences"})
        self.assertEqual(result["sentences"][0]["evidences"][0]["para_id"], 1)


if __name__ == "__main__":
    unittest.main()
