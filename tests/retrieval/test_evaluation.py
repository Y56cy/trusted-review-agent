import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "evaluate_retrieval", ROOT / "scripts/evaluate_retrieval.py"
)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.paragraphs = [{"para_id": i, "page": 1, "text": f"中文内容{i}"} for i in (1, 2, 3)]
        self.queries = [
            {"query_id": "q1", "query": "中文", "relevant_para_ids": [1, 3], "split": "dev"}
        ]

    def test_hand_calculated_metrics(self):
        scores = evaluation.compute_metrics([2, 3], [1, 3], 3)
        self.assertEqual(scores, {"precision": 1 / 3, "recall": 1 / 2, "mrr": 1 / 2})
        self.assertEqual(
            evaluation.compute_metrics([], [1], 3), {"precision": 0, "recall": 0, "mrr": 0}
        )
        self.assertEqual(evaluation.compute_metrics([1, 3, 2], [1, 3], 1)["recall"], 0.5)
        with self.assertRaises(ValueError):
            evaluation.compute_metrics([1, 1], [1], 3)

    def test_invalid_annotation(self):
        evaluation.validate_dataset(self.paragraphs, self.queries)
        for changes in (
            {"query_id": ""},
            {"relevant_para_ids": []},
            {"relevant_para_ids": [99]},
            {"relevant_para_ids": [1, 1]},
            {"relevant_para_ids": [True]},
            {"query": " "},
            {"split": "test"},
        ):
            q = {**self.queries[0], **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                evaluation.validate_dataset(self.paragraphs, [q])
        for paragraphs, queries in (
            ([], self.queries),
            (self.paragraphs, []),
            (self.paragraphs * 2, self.queries),
            (self.paragraphs, self.queries * 2),
        ):
            with self.assertRaises(ValueError):
                evaluation.validate_dataset(paragraphs, queries)

    def test_evaluation_uses_rank_order_and_macro_average(self):
        class Stub:
            def search(self, query, **kwargs):
                return [
                    {"para_id": i, "page": 1, "text": "中文", "score": score}
                    for i, score in ((2, 0.99), (3, 0.1))
                ]

        summary, details = evaluation.evaluate(Stub(), self.queries, 3, 0.3)
        self.assertEqual(len(details), 3)
        self.assertEqual(summary["dev"]["hybrid"]["mrr"], 0.5)
        self.assertEqual(summary["heldout"], {})
        self.assertTrue(details[0]["ranking"][1]["hit"])

    def test_example_counts_splits_and_labels(self):
        ps = evaluation.load_json(ROOT / "data/retrieval/paragraphs.example.json")
        qs = evaluation.load_json(ROOT / "data/retrieval/queries.example.json")
        evaluation.validate_dataset(ps["paragraphs"], qs["queries"])
        self.assertEqual(len(ps["paragraphs"]), 20)
        self.assertEqual(sum(q["split"] == "dev" for q in qs["queries"]), 4)
        self.assertEqual(sum(q["split"] == "heldout" for q in qs["queries"]), 8)
        self.assertEqual(ps["dataset_type"], "synthetic_example")

    def test_validation_only_needs_no_model(self):
        self.assertEqual(
            evaluation.main(
                [
                    "--paragraphs",
                    str(ROOT / "data/retrieval/paragraphs.example.json"),
                    "--queries",
                    str(ROOT / "data/retrieval/queries.example.json"),
                    "--validate-only",
                ]
            ),
            0,
        )

    def test_file_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                evaluation.sha256(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )
            self.assertEqual(
                evaluation.model_manifest(directory), {"model.bin": evaluation.sha256(path)}
            )

    def test_report_records_decline_and_split_metrics(self):
        details = [
            {
                **self.queries[0],
                "method": "vector",
                "metrics": {"precision": 1 / 3, "recall": 0.5, "mrr": 1},
            },
            {
                **self.queries[0],
                "method": "hybrid",
                "metrics": {"precision": 0, "recall": 0, "mrr": 0},
            },
        ]
        text = evaluation.markdown_report(
            {
                "metadata": {
                    "notice": evaluation.NOTICE,
                    "created_at": "fixture",
                    "corpus_sha256": "abc",
                    "queries_sha256": "def",
                    "model_id": "fixture",
                    "top_k": 3,
                    "min_chinese_ratio": 0.3,
                    "model_load_seconds": 0,
                    "index_seconds": 0,
                },
                "summary": {
                    "dev": {
                        "hybrid": {
                            "count": 1,
                            "precision": 0,
                            "recall": 0,
                            "mrr": 0,
                            "mean_query_seconds": 0,
                        },
                        "vector": {
                            "count": 1,
                            "precision": 1 / 3,
                            "recall": 0.5,
                            "mrr": 1,
                            "mean_query_seconds": 0,
                        },
                    }
                },
                "details": details,
            }
        )
        self.assertIn("-0.5000", text)
        self.assertIn("q1 (dev)", text)
        self.assertIn(evaluation.NOTICE, text)


if __name__ == "__main__":
    unittest.main()
