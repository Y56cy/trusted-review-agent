"""多文献语料快照、相关性标注校验与本地标注材料。只依赖标准库。"""

import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paragraphs_checksum(paragraphs):
    return hashlib.sha256(
        json.dumps(paragraphs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def paragraph_key(paragraph):
    return paragraph.get("source", ""), paragraph["para_id"]


def reference(key):
    return {"source": key[0], "para_id": key[1]}


def validate_paragraphs(paragraphs, *, require_sources=False):
    if not isinstance(paragraphs, list) or not paragraphs:
        raise ValueError("paragraphs 必须是非空列表")
    result, ids = [], set()
    for p in paragraphs:
        if not isinstance(p, dict) or not isinstance(p.get("text"), str) or not p["text"].strip():
            raise ValueError("段落 text 必须是非空字符串")
        for field in ("page", "para_id"):
            if type(p.get(field)) is not int or p[field] <= 0:
                raise ValueError(f"段落 {field} 必须是正整数")
        source = p.get("source", "")
        if not isinstance(source, str) or (source and not source.strip()):
            raise ValueError("source 必须是有效文献名字符串")
        key = (source, p["para_id"])
        if key in ids:
            raise ValueError(f"重复段落标识：{key}")
        ids.add(key)
        result.append({**p, "source": source})
    sources = {p["source"] for p in result}
    if "" in sources and (require_sources or len(sources) > 1):
        raise ValueError("多文献或真实语料必须为所有段落提供非空 source")
    return result


def normalize_queries(paragraphs, queries):
    """返回副本，旧编号只在整库唯一时解析，绝不猜测 source。"""
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries 必须是非空列表")
    ids = {paragraph_key(p) for p in paragraphs}
    by_number = defaultdict(list)
    for key in ids:
        by_number[key[1]].append(key)
    result, seen = [], set()
    for q in queries:
        if not isinstance(q, dict):
            raise ValueError("query 条目必须为对象")
        qid = q.get("query_id")
        if not isinstance(qid, str) or not qid.strip() or qid in seen:
            raise ValueError("query_id 必须为唯一非空字符串")
        seen.add(qid)
        if not isinstance(q.get("query"), str) or not q["query"].strip():
            raise ValueError(f"{qid}: query 不能为空")
        if q.get("split") not in ("dev", "heldout"):
            raise ValueError(f"{qid}: split 必须为 dev 或 heldout")
        if ("relevant_paragraphs" in q) == ("relevant_para_ids" in q):
            raise ValueError(f"{qid}: 必须且只能提供一种相关性字段")
        keys = []
        if "relevant_paragraphs" in q:
            refs = q["relevant_paragraphs"]
            if not isinstance(refs, list) or not refs:
                raise ValueError(f"{qid}: relevant_paragraphs 必须是非空列表")
            for ref in refs:
                if (
                    not isinstance(ref, dict)
                    or not isinstance(ref.get("source"), str)
                    or not ref["source"].strip()
                    or type(ref.get("para_id")) is not int
                    or ref["para_id"] <= 0
                ):
                    raise ValueError(f"{qid}: 相关段落须包含非空 source 和正整数 para_id")
                keys.append((ref["source"], ref["para_id"]))
        else:
            numbers = q["relevant_para_ids"]
            if (
                not isinstance(numbers, list)
                or not numbers
                or any(type(i) is not int or i <= 0 for i in numbers)
            ):
                raise ValueError(f"{qid}: relevant_para_ids 必须是非空正整数列表")
            for number in numbers:
                candidates = by_number[number]
                if len(candidates) > 1:
                    raise ValueError(
                        f"{qid}: 段号 {number} 跨文献存在歧义，请使用 relevant_paragraphs"
                    )
                if not candidates:
                    raise ValueError(f"{qid}: 相关段落不存在：{number}")
                keys.append(candidates[0])
        if len(set(keys)) != len(keys) or not set(keys) <= ids:
            raise ValueError(f"{qid}: 相关段落重复或不存在于语料中")
        normalized = {k: v for k, v in q.items() if k != "relevant_para_ids"}
        normalized["relevant_paragraphs"] = [reference(key) for key in keys]
        result.append(normalized)
    return result


def validate_dataset(paragraphs, queries):
    return normalize_queries(validate_paragraphs(paragraphs), queries)


def document_manifest(paragraphs, originals=None):
    groups = defaultdict(list)
    for p in paragraphs:
        groups[p["source"]].append(p)
    originals = {d["source"]: d for d in originals or []}
    return [
        {
            **originals.get(source, {}),
            "source": source,
            "paragraph_count": len(group),
            "parsed_paragraphs_sha256": paragraphs_checksum(group),
        }
        for source, group in sorted(groups.items())
    ]


def load_corpus(*, pdf_paths=None, paragraphs_path=None):
    """返回段落、类型和清单。PDF 全部成功解析后才返回，不跳过失败文件。"""
    if pdf_paths:
        paths = [Path(p).resolve() for p in pdf_paths]
        stems = [p.stem.casefold() for p in paths]
        if len(set(stems)) != len(stems):
            raise ValueError("PDF 文件名主体重复，会造成 source 冲突，请重命名后重试")
        from parser import parse_pdf

        paragraphs, originals = [], []
        for path in paths:
            before = sha256(path)
            try:
                parsed = parse_pdf(str(path))
            except Exception as exc:
                raise ValueError(f"PDF 解析失败 {path.name}：{exc}") from exc
            if not parsed:
                raise ValueError(f"PDF 无有效段落：{path.name}，请使用可提取文字的文本型 PDF")
            if sha256(path) != before:
                raise ValueError(f"PDF 在解析期间发生变化：{path.name}")
            if any(p.get("source") != path.stem for p in parsed):
                raise ValueError(f"解析器 source 与文件名不一致：{path.name}")
            paragraphs.extend(parsed)
            originals.append({"source": path.stem, "filename": path.name, "pdf_sha256": before})
        paragraphs = validate_paragraphs(paragraphs, require_sources=True)
        kind = "unverified_pdf"
    else:
        payload = load_json(paragraphs_path)
        kind = (
            payload.get("dataset_type", "unverified") if isinstance(payload, dict) else "unverified"
        )
        raw = payload["paragraphs"] if isinstance(payload, dict) else payload
        paragraphs = validate_paragraphs(
            raw, require_sources=kind in ("real", "real_corpus", "unverified_pdf")
        )
        embedded = payload.get("manifest", {}) if isinstance(payload, dict) else {}
        if not isinstance(embedded, dict) or not isinstance(embedded.get("documents", []), list):
            raise ValueError("manifest 必须是对象，documents 必须是列表")
        originals = embedded.get("documents", [])
        if any(
            not isinstance(d, dict)
            or not isinstance(d.get("source"), str)
            or type(d.get("paragraph_count")) is not int
            or not isinstance(d.get("parsed_paragraphs_sha256"), str)
            for d in originals
        ):
            raise ValueError("文献清单缺少有效 source、paragraph_count 或 parsed_paragraphs_sha256")
        expected = payload.get("parsed_paragraphs_sha256") if isinstance(payload, dict) else None
        if expected and expected != paragraphs_checksum(paragraphs):
            raise ValueError("语料快照校验值不匹配，语料发生变化，必须重新核对标注")
        if isinstance(payload, dict) and payload.get("manifest"):
            manifest = payload["manifest"]
            if manifest.get("parsed_paragraphs_sha256") != paragraphs_checksum(paragraphs):
                raise ValueError("清单中的语料校验值与段落不一致")
            actual = document_manifest(paragraphs)
            declared = [
                (d["source"], d["paragraph_count"], d["parsed_paragraphs_sha256"])
                for d in originals
            ]
            if sorted(declared) != [
                (d["source"], d["paragraph_count"], d["parsed_paragraphs_sha256"]) for d in actual
            ]:
                raise ValueError("清单中的文献段落数或校验值不一致")
    manifest = {
        "parsed_paragraphs_sha256": paragraphs_checksum(paragraphs),
        "paragraph_count": len(paragraphs),
        "documents": document_manifest(paragraphs, originals),
        "provenance_notice": "JSON 内原始 PDF 校验值为声明值；使用提供的原 PDF 可重新核对。",
    }
    return paragraphs, kind, manifest


def prepare_materials(paragraphs, kind, manifest, output_dir):
    output_dir = Path(output_dir)
    names = ("corpus.json", "corpus-manifest.json", "paragraphs.html", "queries.template.json")
    if any((output_dir / name).exists() for name in names):
        raise ValueError("准备材料输出目录已有快照或模板，请换新目录，避免覆盖已标注资料")
    template = {
        "dataset_type": kind,
        "annotation_status": "draft",
        "annotator": "",
        "reviewer": "",
        "review_notes": "待填写；需要另一成员独立看原文复核，不能把初稿当成标准答案。",
        "corpus_sha256": manifest["parsed_paragraphs_sha256"],
        "instructions": "目标20条：6 dev/14 heldout；仅标注原文可回答的问题；边界问题另存。",
        "query_format": {
            "query_id": "q001",
            "query": "待填写",
            "split": "dev",
            "relevant_paragraphs": [{"source": "文献名", "para_id": 1}],
            "annotation_status": "draft",
            "review_notes": "标注依据和分歧处理",
        },
        "queries": [],
    }
    blocks = []
    for p in sorted(paragraphs, key=lambda p: (p["source"], p["page"], p["para_id"])):
        blocks.append(
            f"<article><h2>{html.escape(p['source'] or '旧单篇语料')} · "
            f"第{p['page']}页 · 第{p['para_id']}段</h2><p>{html.escape(p['text'])}</p></article>"
        )
    html_text = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        "<title>段落标注查看材料</title><style>body{max-width:1000px;margin:30px auto;"
        "font-family:system-ui;line-height:1.8}article{border-bottom:1px solid #ccc;"
        "padding:12px}p{white-space:pre-wrap;overflow-wrap:anywhere}h2{font-size:18px}</style>"
        "<h1>段落标注查看材料</h1><p>请对照原 PDF 核对页码与文字。"
        "本材料不包含检索排名或自动相关性标签。</p>" + "".join(blocks) + "</html>"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "corpus.json": {
            "dataset_type": kind,
            "parsed_paragraphs_sha256": manifest["parsed_paragraphs_sha256"],
            "manifest": manifest,
            "paragraphs": paragraphs,
        },
        "corpus-manifest.json": manifest,
        "queries.template.json": template,
    }
    for name, data in outputs.items():
        (output_dir / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "paragraphs.html").write_text(html_text, encoding="utf-8")


def annotation_review(payload, queries, corpus_checksum):
    """只核验复核声明的完整性，不声称程序能证明人工复核实际发生。"""
    header = payload if isinstance(payload, dict) else {}
    expected = header.get("corpus_sha256")
    if expected is not None and (not isinstance(expected, str) or not expected.strip()):
        if expected != "":
            raise ValueError("corpus_sha256 必须是字符串校验值")
    if expected and expected != corpus_checksum:
        raise ValueError("查询绑定的语料校验值不匹配，必须重新核对标注")
    reviews = []
    for q in queries:
        review = {
            name: q.get(name, header.get(name, ""))
            for name in ("annotation_status", "annotator", "reviewer", "review_notes")
        }
        if any(not isinstance(value, str) for value in review.values()):
            raise ValueError("标注状态、标注者、复核者和复核说明必须为字符串")
        review["query_id"] = q["query_id"]
        review["complete"] = (
            review["annotation_status"] == "reviewed"
            and all(
                isinstance(review[name], str) and review[name].strip()
                for name in ("annotator", "reviewer", "review_notes")
            )
            and review["annotator"].strip() != review["reviewer"].strip()
        )
        reviews.append(review)
    complete = bool(expected) and all(r["complete"] for r in reviews)
    return {
        "status": "reviewed_declared" if complete else "preliminary",
        "corpus_bound": bool(expected),
        "queries": reviews,
        "counts": dict(Counter(r["annotation_status"] or "unspecified" for r in reviews)),
        "notice": (
            "复核信息由标注者声明，程序不代替人工复核；"
            "未绑定语料或复核信息不全仅为初步评测。"
        ),
    }
