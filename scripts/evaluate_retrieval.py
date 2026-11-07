"""独立检索对照评测。不要通过发现 tests/ 下的历史 spike 运行评测。"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
METHODS = ("vector", "bm25", "hybrid")
NOTICE = (
    "仅验证检索功能与评测流程，不能代表真实科研论文表现。"
    "示例相关性标签为预先编写，尚未经过独立人工复核。"
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset(paragraphs, queries):
    """模型加载前校验；编号限定为 parser 产生的正整数。"""
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("paragraphs 必须是非空列表")
    ids = set()
    for p in paragraphs:
        if not isinstance(p, dict) or not isinstance(p.get("text"), str) or not p["text"].strip():
            raise ValueError("段落 text 必须是非空字符串")
        for key in ("para_id", "page"):
            if type(p.get(key)) is not int or p[key] <= 0:
                raise ValueError(f"段落 {key} 必须是正整数")
        if p["para_id"] in ids:
            raise ValueError(f"重复 para_id：{p['para_id']}")
        ids.add(p["para_id"])
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries 必须是非空列表")
    query_ids = set()
    for q in queries:
        if not isinstance(q, dict):
            raise ValueError("query 条目必须为对象")
        qid = q.get("query_id")
        if not isinstance(qid, str) or not qid.strip() or qid in query_ids:
            raise ValueError("query_id 必须为唯一非空字符串")
        query_ids.add(qid)
        if not isinstance(q.get("query"), str) or not q["query"].strip():
            raise ValueError(f"{qid}: query 不能为空")
        if q.get("split") not in ("dev", "heldout"):
            raise ValueError(f"{qid}: split 必须为 dev 或 heldout")
        relevant = q.get("relevant_para_ids")
        if (
            not isinstance(relevant, list)
            or not relevant
            or any(type(i) is not int or i <= 0 for i in relevant)
        ):
            raise ValueError(f"{qid}: relevant_para_ids 必须是非空正整数列表")
        if len(set(relevant)) != len(relevant) or not set(relevant) <= ids:
            raise ValueError(f"{qid}: 相关编号重复或不存在于语料中")


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
    details = []
    for q in queries:
        for method in METHODS:
            start = perf_counter()
            results = retriever.search(
                q["query"], top_k=top_k, min_chinese_ratio=min_chinese_ratio, method=method
            )
            elapsed = perf_counter() - start
            relevant = set(q["relevant_para_ids"])
            details.append(
                {
                    **q,
                    "method": method,
                    "query_seconds": elapsed,
                    "metrics": compute_metrics([r["para_id"] for r in results], relevant, top_k),
                    "ranking": [
                        {**r, "rank": rank, "hit": r["para_id"] in relevant}
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


def markdown_report(report):
    m = report["metadata"]
    lines = [
        "# 检索对照评测",
        "",
        m["notice"],
        "",
        f"- UTC 时间：{m['created_at']}",
        f"- 语料 SHA-256：`{m['corpus_sha256']}`",
        f"- 查询 SHA-256：`{m['queries_sha256']}`",
        f"- 模型：`{m['model_id']}`（文件校验值见同目录 JSON）",
        f"- 参数：k={m['top_k']}，中文占比={m['min_chinese_ratio']}；"
        "BM25 k1=1.5、b=0.75；RRF 常数=60、双通道权重各 0.5",
        f"- 模型加载：{m['model_load_seconds']:.3f}s；建索引：{m['index_seconds']:.3f}s",
        "- score 始终是余弦相似度；指标按返回顺序计算。",
        "",
        "耗时是本机顺序执行的观察值，包含首次调用开销，不作为性能承诺。",
        "",
    ]
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
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--paragraphs", type=Path)
    source.add_argument("--pdf", type=Path)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--min-chinese-ratio", type=float, default=0.3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/retrieval")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.top_k <= 0 or not 0 <= args.min_chinese_ratio <= 1:
        parser.error("top-k 必须为正数，min-chinese-ratio 必须在 0 到 1 之间")
    try:
        if args.pdf:
            from parser import parse_pdf

            paragraphs = parse_pdf(str(args.pdf))
            corpus_kind = "unverified_pdf"
        else:
            payload = load_json(args.paragraphs)
            corpus_kind = (
                payload.get("dataset_type", "unverified")
                if isinstance(payload, dict)
                else "unverified"
            )
            paragraphs = payload["paragraphs"] if isinstance(payload, dict) else payload
        query_payload = load_json(args.queries)
        queries = query_payload["queries"] if isinstance(query_payload, dict) else query_payload
        validate_dataset(paragraphs, queries)
        if args.validate_only:
            print(
                f"校验通过：{len(paragraphs)} 段，{len(queries)} 条查询。未加载模型、未运行效果评测。"
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
            "bm25": {"k1": 1.5, "b": 0.75},
            "rrf": {"k": 60, "weights": [0.5, 0.5]},
            "candidate_limit": "min(eligible_count, max(20, top_k))",
            "tokenizer": "jieba default dictionary; HMM=False; lowercase; no stopwords",
            "corpus_sha256": sha256(args.pdf or args.paragraphs),
            "parsed_paragraphs_sha256": hashlib.sha256(
                json.dumps(paragraphs, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "queries_sha256": sha256(args.queries),
            "model_id": "BAAI/bge-small-zh-v1.5",
            "model_path": str(model_path),
            "model_files_sha256": model_manifest(model_path),
            "model_load_seconds": load_seconds,
            "index_seconds": index_seconds,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": packages,
        }
        report = {"metadata": metadata, "summary": summary, "details": details}
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
