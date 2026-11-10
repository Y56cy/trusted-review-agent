"""多文献评测回归；临时 PDF/模型替身只验证流程，不代表真实论文效果。"""

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_retrieval as ev
import retrieval_data as data


class MultiSourceTests(unittest.TestCase):
    def setUp(self):
        self.paragraphs = [
            {"source": "甲", "para_id": 1, "page": 1, "text": "甲文献讨论检索证据的准确性。"},
            {"source": "乙", "para_id": 1, "page": 2, "text": "乙文献讨论不同的检索方法。"},
        ]
        self.query = {
            "query_id": "q1",
            "query": "检索",
            "split": "dev",
            "relevant_paragraphs": [{"source": "甲", "para_id": 1}],
        }

    def test_compound_identity_and_wrong_source_not_hit(self):
        queries = ev.validate_dataset(self.paragraphs, [self.query])

        class Stub:
            def search(inner, query, **kwargs):
                return [{**self.paragraphs[1], "score": 0.9}]

        summary, details = ev.evaluate(Stub(), queries, 3, 0.3)
        self.assertEqual(summary["all"]["hybrid"]["recall"], 0)
        self.assertFalse(details[0]["ranking"][0]["hit"])

    def test_cross_document_metrics_hand_calculated(self):
        scores = ev.compute_metrics([("乙", 1), ("甲", 1)], [("甲", 1), ("丙", 2)], 3)
        self.assertEqual(scores, {"precision": 1 / 3, "recall": 0.5, "mrr": 0.5})

    def test_invalid_refs_and_sources(self):
        for refs in (
            [],
            [{"source": "丙", "para_id": 1}],
            [{"source": "甲", "para_id": 1}] * 2,
            [{"source": "", "para_id": 1}],
            [{"source": [], "para_id": 1}],
            [{"source": "甲", "para_id": True}],
        ):
            with self.subTest(refs=refs), self.assertRaises(ValueError):
                ev.validate_dataset(self.paragraphs, [{**self.query, "relevant_paragraphs": refs}])
        with self.assertRaises(ValueError):
            ev.validate_dataset(self.paragraphs * 2, [self.query])
        for source in ([], 3, " ", None, ""):
            paragraphs = copy.deepcopy(self.paragraphs)
            paragraphs[0]["source"] = source
            with self.subTest(source=source), self.assertRaises(ValueError):
                ev.validate_dataset(paragraphs, [self.query])

    def test_legacy_ids_only_when_unambiguous(self):
        old = {k: v for k, v in self.query.items() if k != "relevant_paragraphs"}
        old["relevant_para_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "歧义"):
            ev.validate_dataset(self.paragraphs, [old])
        normalized = ev.validate_dataset(self.paragraphs[:1], [old])
        self.assertEqual(normalized[0]["relevant_paragraphs"], self.query["relevant_paragraphs"])
        self.assertIn("relevant_para_ids", old)
        with self.assertRaises(ValueError):
            ev.validate_dataset(self.paragraphs, [{**self.query, "relevant_para_ids": [1]}])

    def test_duplicate_stems_rejected_before_parse(self):
        with patch("parser.parse_pdf") as parse, self.assertRaisesRegex(ValueError, "主体重复"):
            data.load_corpus(pdf_paths=[Path("one/Paper.pdf"), Path("two/paper.pdf")])
        parse.assert_not_called()

    def test_prepare_no_model_escaped_html_and_no_overwrite(self):
        import retriever

        paragraphs = copy.deepcopy(self.paragraphs)
        paragraphs[0]["text"] = "中文证据<script>alert(1)</script>"
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            source = target / "input.json"
            source.write_text(
                json.dumps({"dataset_type": "real_corpus", "paragraphs": paragraphs}), "utf-8"
            )
            out = target / "materials"
            with patch.object(retriever, "_load_embedding_model") as loader:
                self.assertEqual(
                    ev.main(
                        ["--paragraphs", str(source), "--prepare-only", "--output-dir", str(out)]
                    ),
                    0,
                )
            loader.assert_not_called()
            html = (out / "paragraphs.html").read_text("utf-8")
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            template = data.load_json(out / "queries.template.json")
            self.assertEqual(template["queries"], [])
            loaded, kind, manifest = data.load_corpus(paragraphs_path=out / "corpus.json")
            self.assertEqual(loaded, paragraphs)
            self.assertEqual(len(manifest["documents"]), 2)
            with self.assertRaises(ValueError):
                data.prepare_materials(loaded, kind, manifest, out)
            payload = data.load_json(out / "corpus.json")
            payload["paragraphs"][0]["text"] += "变更"
            (out / "changed.json").write_text(json.dumps(payload), "utf-8")
            with self.assertRaisesRegex(ValueError, "校验值"):
                data.load_corpus(paragraphs_path=out / "changed.json")

    def test_review_declaration_requires_binding_and_distinct_people(self):
        checksum = data.paragraphs_checksum(self.paragraphs)
        header = {
            "corpus_sha256": checksum,
            "annotation_status": "reviewed",
            "annotator": "成员A",
            "reviewer": "成员B",
            "review_notes": "已看原文确认",
        }
        self.assertEqual(
            data.annotation_review(header, [self.query], checksum)["status"], "reviewed_declared"
        )
        for bad in (
            {"reviewer": "成员A"},
            {"reviewer": ""},
            {"review_notes": ""},
            {"annotation_status": "draft"},
            {"corpus_sha256": ""},
        ):
            self.assertEqual(
                data.annotation_review({**header, **bad}, [self.query], checksum)["status"],
                "preliminary",
            )
        with self.assertRaisesRegex(ValueError, "校验值"):
            data.annotation_review(header, [self.query], "wrong")
        with self.assertRaisesRegex(ValueError, "字符串"):
            data.annotation_review({**header, "annotation_status": []}, [self.query], checksum)

    def test_case_analysis_reports_categories_without_inventing_causes(self):
        rows = []
        for qid, base, hybrid in (("better", 0, 1), ("same", 1, 1), ("worse", 1, 0)):
            for method, score in (("vector", base), ("hybrid", hybrid)):
                rows.append(
                    {
                        "query_id": qid,
                        "query": "问题",
                        "split": "dev",
                        "method": method,
                        "metrics": {"precision": score, "recall": score, "mrr": score},
                        "ranking": [{"source": "甲", "para_id": 1}] if score else [],
                    }
                )
        groups = ev.analyze_cases(rows)
        self.assertEqual(groups["improved"][0]["query_id"], "better")
        self.assertEqual(groups["unchanged"][0]["query_id"], "same")
        self.assertEqual(groups["declined"][0]["query_id"], "worse")
        self.assertEqual(groups["missed"][0]["query_id"], "worse")
        self.assertIn("待人工", groups["missed"][0]["cause"])

    def test_existing_report_is_not_overwritten_and_missing_queries_rejected(self):
        import retriever

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "report.json").write_text("keep previous experiment", "utf-8")
            with (
                patch.object(retriever, "_load_embedding_model") as loader,
                self.assertRaises(SystemExit),
            ):
                ev.main(
                    [
                        "--paragraphs",
                        str(ROOT / "data/retrieval/paragraphs.example.json"),
                        "--queries",
                        str(ROOT / "data/retrieval/queries.example.json"),
                        "--output-dir",
                        str(root),
                    ]
                )
            loader.assert_not_called()
            self.assertEqual((root / "report.json").read_text("utf-8"), "keep previous experiment")
        with self.assertRaises(SystemExit):
            ev.main(["--paragraphs", "unused.json"])

    def test_invalid_manifest_is_rejected_without_attribute_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            for manifest in (None, [], {"documents": None}, {"documents": [{}]}):
                path.write_text(
                    json.dumps({"paragraphs": self.paragraphs, "manifest": manifest}), "utf-8"
                )
                with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                    data.load_corpus(paragraphs_path=path)

    def test_filtered_relevance_stays_in_denominator(self):
        from test_retriever import FakeEncoder

        import retriever

        paragraphs = [
            self.paragraphs[0],
            {**self.paragraphs[1], "text": "English references only."},
        ]
        query = {
            **self.query,
            "relevant_paragraphs": [{"source": "甲", "para_id": 1}, {"source": "乙", "para_id": 1}],
        }
        with patch.object(retriever, "_load_embedding_model", return_value=FakeEncoder()):
            r = retriever.Retriever()
        r.build_index(paragraphs)
        summary, _ = ev.evaluate(r, ev.validate_dataset(paragraphs, [query]), 3, 0.3)
        self.assertEqual(summary["all"]["vector"]["recall"], 0.5)

    @unittest.skipUnless(importlib.util.find_spec("pypdf"), "需 pypdf")
    def test_multi_pdf_prepare_and_snapshot_evaluation(self):
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
        from test_retriever import FakeEncoder

        import retriever

        def write_pdf(path):
            writer = PdfWriter()
            page = writer.add_blank_page(width=600, height=800)
            font = DictionaryObject(
                {
                    NameObject("/Type"): NameObject("/Font"),
                    NameObject("/Subtype"): NameObject("/Type1"),
                    NameObject("/BaseFont"): NameObject("/Helvetica"),
                }
            )
            page[NameObject("/Resources")] = DictionaryObject(
                {
                    NameObject("/Font"): DictionaryObject(
                        {NameObject("/F1"): writer._add_object(font)}
                    )
                }
            )
            stream = DecodedStreamObject()
            stream.set_data(
                b"BT /F1 12 Tf 40 700 Td (BM25 research evidence retrieval uses term frequency.) Tj ET"
            )
            page[NameObject("/Contents")] = writer._add_object(stream)
            with path.open("wb") as output:
                writer.write(output)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "甲.pdf", root / "乙.pdf"
            write_pdf(a)
            write_pdf(b)
            out = root / "materials"
            self.assertEqual(
                ev.main(["--pdf", str(a), str(b), "--prepare-only", "--output-dir", str(out)]), 0
            )
            template = data.load_json(out / "queries.template.json")
            template["queries"] = [{**self.query, "query": "BM25"}]
            queries = root / "queries.json"
            queries.write_text(json.dumps(template), "utf-8")
            with patch.object(
                retriever, "_load_embedding_model", return_value=FakeEncoder()
            ) as load:
                self.assertEqual(
                    ev.main(
                        [
                            "--paragraphs",
                            str(out / "corpus.json"),
                            "--queries",
                            str(queries),
                            "--min-chinese-ratio",
                            "0",
                            "--output-dir",
                            str(root / "report"),
                        ]
                    ),
                    0,
                )
            self.assertEqual(load.call_count, 1)
            report = data.load_json(root / "report/report.json")
            self.assertEqual(report["metadata"]["evaluation_status"], "preliminary")
            self.assertEqual(len(report["metadata"]["corpus_manifest"]["documents"]), 2)
            for row in report["details"]:
                self.assertEqual(sum(r["hit"] for r in row["ranking"]), 1)
                self.assertEqual(len(row["ranking"]), 2)
            md = (root / "report/report.md").read_text("utf-8")
            self.assertIn("初步评测", md)
            self.assertIn("第1段", md)
            self.assertIn("原文：", md)
            # 同一组 PDF 直接评测也应成功，且仍只加载一次模型。
            with patch.object(
                retriever, "_load_embedding_model", return_value=FakeEncoder()
            ) as load:
                self.assertEqual(
                    ev.main(
                        [
                            "--pdf",
                            str(a),
                            str(b),
                            "--queries",
                            str(queries),
                            "--min-chinese-ratio",
                            "0",
                            "--output-dir",
                            str(root / "direct"),
                        ]
                    ),
                    0,
                )
            self.assertEqual(load.call_count, 1)

    @unittest.skipUnless(importlib.util.find_spec("pypdf"), "需 pypdf")
    def test_empty_and_invalid_pdf_do_not_silently_skip(self):
        from pypdf import PdfWriter

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty = root / "blank.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=200, height=200)
            with empty.open("wb") as output:
                writer.write(output)
            with self.assertRaisesRegex(ValueError, "无有效段落"):
                data.load_corpus(pdf_paths=[empty])
            invalid = root / "invalid.pdf"
            invalid.write_bytes(b"not a PDF")
            with self.assertRaisesRegex(ValueError, "解析失败"):
                data.load_corpus(pdf_paths=[invalid])


if __name__ == "__main__":
    unittest.main()
