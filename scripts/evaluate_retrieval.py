"""独立检索对照评测。不要通过发现 tests/ 下的历史 spike 运行评测。"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
METHODS = ("vector", "bm25", "hybrid")
NOTICE = (
    "仅验证检索功能与评测流程，不能代表真实科研论文表现。"
    "示例相关性标签为预先编写，尚未经过独立人工复核。"
)


from retrieval_data import (  # noqa: E402
    annotation_review,
    load_corpus,
    load_json,
    normalize_queries,
    paragraph_key,
    prepare_materials,
    sha256,
    validate_dataset,
)


def model_identifier(model_path, default_path):
    """自定义目录不伪称默认 BGE；具体文件身份由报告校验值确定。"""
    if model_path.resolve() == default_path.resolve():
        return "BAAI/bge-small-zh-v1.5"
    return f"local:{model_path.name}"


def compute_metrics(ranked_ids, relevant_ids, top_k):
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k 必须为正整数")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("相关段落不能为空")
    ranked = ranked_ids[:top_k]
    if len(set(ranked)) != len(ranked):
        raise ValueError("结果包含重复段落")
    hits = set(ranked) & relevant
    reciprocal_rank = next((1 / rank for rank, i in enumerate(ranked, 1) if i in relevant), 0.0)
    return {
        "precision": len(hits) / top_k,
        "recall": len(hits) / len(relevant),
        "mrr": reciprocal_rank,
    }


def evaluate(retriever, queries, top_k, min_chinese_ratio):
    """三种模式共用同一模型和索引，保留每种模式的结果次序。"""
    if any("relevant_paragraphs" not in q for q in queries):
        corpus = getattr(retriever, "corpus", None)
        if corpus is None:
            # Compatibility for standalone legacy callers without a corpus attribute.
            corpus = [
                {"para_id": i}
                for i in sorted({i for q in queries for i in q.get("relevant_para_ids", [])})
            ]
        queries = normalize_queries(corpus, queries)
    details = []
    for q in queries:
        for method in METHODS:
            start = perf_counter()
            results = retriever.search(
                q["query"], top_k=top_k, min_chinese_ratio=min_chinese_ratio, method=method
            )
            elapsed = perf_counter() - start
            relevant = {paragraph_key(ref) for ref in q["relevant_paragraphs"]}
            details.append(
                {
                    **q,
                    "method": method,
                    "query_seconds": elapsed,
                    "metrics": compute_metrics(
                        [paragraph_key(r) for r in results], relevant, top_k
                    ),
                    "ranking": [
                        {**r, "rank": rank, "hit": paragraph_key(r) in relevant}
                        for rank, r in enumerate(results, 1)
                    ],
                }
            )
    summary = {}
    for split in ("all", "dev", "heldout"):
        summary[split] = {}
        for method in METHODS:
            rows = [
                d
                for d in details
                if d["method"] == method and (split == "all" or d["split"] == split)
            ]
            if rows:
                summary[split][method] = {
                    "count": len(rows),
                    **{
                        metric: sum(d["metrics"][metric] for d in rows) / len(rows)
                        for metric in ("precision", "recall", "mrr")
                    },
                    "mean_query_seconds": sum(d["query_seconds"] for d in rows) / len(rows),
                }
    return summary, details


def model_manifest(path):
    """对本地模型文件做内容校验，模型名称相同也可辨别权重差异。"""
    path = Path(path)
    return {
        str(p.relative_to(path)): sha256(p)
        for p in sorted(path.rglob("*"))
        if p.is_file()
        and ".cache" not in p.relative_to(path).parts
        and ".git" not in p.relative_to(path).parts
    }


def markdown_text(text):
    import html

    escaped = html.escape(str(text)).replace("\n", " ").replace("\r", " ")
    for symbol in ("\\", "`", "*", "_", "[", "]", "|", "#"):
        escaped = escaped.replace(symbol, "\\" + symbol)
    return escaped


def analyze_cases(details):
    indexed = {(d["query_id"], d["method"]): d for d in details}
    groups = {name: [] for name in ("improved", "unchanged", "declined", "missed", "no_results")}
    for row in details:
        if row["method"] != "hybrid":
            continue
        base = indexed[(row["query_id"], "vector")]
        delta = {m: row["metrics"][m] - base["metrics"][m] for m in ("precision", "recall", "mrr")}
        case = {
            "query_id": row["query_id"],
            "query": row.get("query", ""),
            "split": row["split"],
            "delta": delta,
            "vector_ranking": base.get("ranking", []),
            "hybrid_ranking": row.get("ranking", []),
            "cause": "待人工对照原文核查，不根据指标自动推断原因",
        }
        if any(d < 0 for d in delta.values()):
            groups["declined"].append(case)
        if any(d > 0 for d in delta.values()):
            groups["improved"].append(case)
        if all(d == 0 for d in delta.values()):
            groups["unchanged"].append(case)
        if row["metrics"]["recall"] < 1:
            groups["missed"].append(case)
        if not row.get("ranking"):
            groups["no_results"].append(case)
    return groups


def markdown_report(report):
    m = report["metadata"]
    preliminary = m.get("evaluation_status", "preliminary") != "reviewed_declared"
    lines = [
        "# 检索对照评测" + ("（初步评测，未完成复核）" if preliminary else "（已声明独立复核）"),
        "",
        m["notice"],
        "",
        f"- UTC 时间：{m['created_at']}",
        f"- 语料 SHA-256：`{m['corpus_sha256']}`",
        f"- 查询 SHA-256：`{m['queries_sha256']}`",
        f"- 模型：`{m['model_id']}`（文件校验值见同目录 JSON）",
        f"- 参数：k={m['top_k']}，中文占比={m['min_chinese_ratio']}；"
        "BM25 k1=1.5、b=0.75；RRF 常数=60、双通道权重各 0.5",
        "- BM25 正值 IDF：ln(1 + (N - df + 0.5) / (df + 0.5))；"
        "关键词同分按段落编号排序，不引入向量分数。",
        f"- 模型加载：{m['model_load_seconds']:.3f}s；建索引：{m['index_seconds']:.3f}s",
        "- score 始终是余弦相似度；指标按返回顺序计算。",
        "",
        "耗时是本机顺序执行的观察值，包含首次调用开销，不作为性能承诺。",
        "",
    ]
    review = m.get("annotation_review", {})
    lines += [
        f"- 复核状态：{review.get('status', 'unspecified')}；"
        "复核信息为人工声明，程序不代替人工确认。",
        "",
        "## 文献清单",
        "",
        "| 文献 source | 段落数 | 原 PDF SHA-256 | 段落 SHA-256 |",
        "|---|---:|---|---|",
    ]
    for document in m.get("corpus_manifest", {}).get("documents", []):
        source = markdown_text(document["source"] or "旧单篇语料")
        lines.append(
            f"| {source} | {document['paragraph_count']} | "
            f"{document.get('pdf_sha256', '未提供')} | {document['parsed_paragraphs_sha256']} |"
        )
    lines.append("")
    for split, methods in report["summary"].items():
        if not methods:
            continue
        lines += [
            f"## {split}",
            "",
            "| 模式 | 查询数 | Precision@k | Recall@k | MRR@k | 平均查询秒数 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for method, row in methods.items():
            lines.append(
                f"| {method} | {row['count']} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['mrr']:.4f} | {row['mean_query_seconds']:.4f} |"
            )
        if "hybrid" in methods:
            delta = methods["hybrid"]["recall"] - methods["vector"]["recall"]
            lines += ["", f"hybrid 相对 vector 的 Recall@k 差值：{delta:+.4f}。"]
        lines.append("")
    lines += ["## 未命中与下降案例", "", "按查询核对 JSON 中的完整排名，不能只看平均分。", ""]
    indexed = {(d["query_id"], d["method"]): d for d in report["details"]}
    found = False
    for d in report["details"]:
        if d["method"] != "hybrid":
            continue
        baseline = indexed[(d["query_id"], "vector")]
        if d["metrics"]["recall"] < 1 or any(
            d["metrics"][key] < baseline["metrics"][key] for key in ("recall", "mrr")
        ):
            found = True
            lines.append(
                f"- {d['query_id']} ({d['split']}): vector={baseline['metrics']}；"
                f"hybrid={d['metrics']}。"
            )
    if not found:
        lines.append(
            "本次未观察到 hybrid 漏召回或相对 vector 的 Recall/MRR 下降；不代表真实语料表现。"
        )
    lines += [
        "",
        "## 案例分类与待核查原因",
        "",
        "原因需区分解析、分词、语义召回、语言过滤、候选截取和标注争议；不可仅凭分数下结论。",
        "",
    ]
    for group, cases in report.get("case_analysis", analyze_cases(report["details"])).items():
        lines.append(
            f"- {group}："
            + (
                "、".join(markdown_text(c["query_id"]) for c in cases)
                if cases
                else "本次不存在此类案例"
            )
        )
    lines += ["", "## 逐查询排名", ""]
    for detail in report["details"]:
        lines += [
            f"### {markdown_text(detail['query_id'])} · {detail['method']} · {detail['split']}",
            "",
            markdown_text(detail.get("query", "")),
            "",
        ]
        if not detail.get("ranking"):
            lines += ["无返回结果。", ""]
        for row in detail.get("ranking", []):
            lines += [
                f"- 排名 {row['rank']}：{markdown_text(row.get('source', '') or '旧单篇语料')} · "
                f"第{row['page']}页 · 第{row['para_id']}段 · "
                f"{'命中' if row['hit'] else '未命中'}；score={row['score']:.6f}",
                f"  原文：{markdown_text(row['text'])}",
                "",
            ]
    lines += ["## 标注复核声明", "", review.get("notice", "未提供复核说明。"), ""]
    for row in review.get("queries", []):
        lines.append(
            f"- {markdown_text(row['query_id'])}: {markdown_text(str(row['annotation_status']))}；"
            f"标注者 {markdown_text(str(row['annotator']))}；"
            f"复核者 {markdown_text(str(row['reviewer']))}；"
            f"说明 {markdown_text(str(row['review_notes']))}"
        )
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--paragraphs", type=Path)
    source.add_argument("--pdf", type=Path, nargs="+")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-chinese-ratio", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/retrieval")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true")
    mode.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--split", choices=("all", "dev", "heldout"), default="all")
    args = parser.parse_args(argv)
    if args.top_k <= 0 or not 0 <= args.min_chinese_ratio <= 1:
        parser.error("top-k 必须为正数，min-chinese-ratio 必须在 0 到 1 之间")
    if not args.prepare_only and args.queries is None:
        parser.error("评测或校验必须提供 --queries；准备标注材料使用 --prepare-only")
    if args.prepare_only and args.queries is not None:
        parser.error("--prepare-only 不读取查询，请省略 --queries")
    try:
        paragraphs, corpus_kind, corpus_manifest = load_corpus(
            pdf_paths=args.pdf, paragraphs_path=args.paragraphs
        )
        if args.prepare_only:
            prepare_materials(paragraphs, corpus_kind, corpus_manifest, args.output_dir)
            print(f"标注材料已写入 {args.output_dir}；未加载模型、未生成评测分数。")
            return 0
        if not args.validate_only and any(
            (args.output_dir / name).exists() for name in ("report.json", "report.md")
        ):
            raise ValueError("输出目录已有评测报告，请指定新的 --output-dir，保留历史实验记录")
        query_payload = load_json(args.queries)
        raw_queries = query_payload["queries"] if isinstance(query_payload, dict) else query_payload
        queries = validate_dataset(paragraphs, raw_queries)
        # 检查全部标签与冻结语料绑定，再选择实际执行的集合。
        annotation_review(query_payload, queries, corpus_manifest["parsed_paragraphs_sha256"])
        if args.split != "all":
            queries = [q for q in queries if q["split"] == args.split]
            if not queries:
                raise ValueError(f"查询文件不包含 {args.split} 查询")
        review = annotation_review(
            query_payload, queries, corpus_manifest["parsed_paragraphs_sha256"]
        )
        if args.validate_only:
            print(
                f"校验通过：{len(paragraphs)} 段，{len(queries)} 条查询。"
                "未加载模型、未运行效果评测。"
            )
            return 0
        from retriever import DEFAULT_MODEL_PATH, Retriever

        model_path = (args.model_path or DEFAULT_MODEL_PATH).resolve()
        start = perf_counter()
        retriever = Retriever(model_path=model_path)
        load_seconds = perf_counter() - start
        start = perf_counter()
        retriever.build_index(paragraphs)
        index_seconds = perf_counter() - start
        summary, details = evaluate(retriever, queries, args.top_k, args.min_chinese_ratio)
        packages = {}
        for name in (
            "jieba",
            "rank-bm25",
            "numpy",
            "torch",
            "transformers",
            "sentence-transformers",
        ):
            try:
                packages[name] = version(name)
            except PackageNotFoundError:
                packages[name] = "not installed"
        synthetic = corpus_kind == "synthetic_example" or (
            isinstance(query_payload, dict)
            and query_payload.get("dataset_type") == "synthetic_example"
        )
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_type": corpus_kind,
            "notice": NOTICE
            if synthetic
            else "外部语料；标签来源及独立人工复核状态需另行确认。检索指标不能证明综述事实正确。",
            "top_k": args.top_k,
            "min_chinese_ratio": args.min_chinese_ratio,
            "bm25": {
                "k1": 1.5,
                "b": 0.75,
                "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))",
                "tie_break": "para_id ascending",
            },
            "requested_split": args.split,
            "rrf": {"k": 60, "weights": [0.5, 0.5]},
            "candidate_limit": "min(eligible_count, max(20, top_k))",
            "tokenizer": "jieba default dictionary; HMM=False; lowercase; no stopwords",
            "corpus_sha256": (
                sha256(args.paragraphs)
                if args.paragraphs
                else sha256(args.pdf[0])
                if len(args.pdf) == 1
                else corpus_manifest["parsed_paragraphs_sha256"]
            ),
            "parsed_paragraphs_sha256": corpus_manifest["parsed_paragraphs_sha256"],
            "corpus_manifest": corpus_manifest,
            "annotation_review": review,
            "evaluation_status": "preliminary"
            if synthetic or review["status"] != "reviewed_declared"
            else "reviewed_declared",
            "queries_sha256": sha256(args.queries),
            "model_id": model_identifier(model_path, DEFAULT_MODEL_PATH),
            "model_path": str(model_path),
            "model_files_sha256": model_manifest(model_path),
            "model_load_seconds": load_seconds,
            "index_seconds": index_seconds,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
            "source_files_sha256": {
                "src/retriever.py": sha256(ROOT / "src/retriever.py"),
                "scripts/evaluate_retrieval.py": sha256(Path(__file__)),
                "scripts/retrieval_data.py": sha256(Path(__file__).with_name("retrieval_data.py")),
                "src/parser.py": sha256(ROOT / "src/parser.py"),
            },
        }
        report = {
            "metadata": metadata,
            "summary": summary,
            "details": details,
            "case_analysis": analyze_cases(details),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
        print(f"报告已写入 {args.output_dir}")
        return 0
    except (ValueError, KeyError, OSError, ImportError) as exc:
        parser.exit(2, f"评测失败：{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
