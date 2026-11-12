"""Member B's local UI: consume the existing pipeline without changing its contract.

Offline imports, rendering and exports use only the standard library. Gradio is
loaded only by build_app(); model code runs in a separate, time-bounded process.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import importlib.metadata
import importlib.util
import json
import math
import os
import re
import secrets
import socket
import subprocess
import sys
import tempfile
import threading
import unicodedata
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
MAX_JSON_BYTES = 5 * 1024 * 1024
MAX_PDF_BYTES = 50 * 1024 * 1024
MAX_PDF_FILES = 10
MAX_TOTAL_PDF_BYTES = 200 * 1024 * 1024
UNKNOWN_SOURCE = "文献来源未知"
NOTICE = "句级匹配仅提供证据支持与人工核查依据，不证明事实正确；NLI 概率不是事实准确率。"
DEMO_NOTICE = "演示数据：手工构造的界面样例，不是论文、真实生成结果或评测结果。"
CHECK_NOTICE = (
    "基础自检不加载模型、不下载文件、不运行生成或正式评测；"
    "即使通过，仍须使用真实 PDF 验证模型加载与完整生成。"
)
CORE_DEPENDENCIES = (
    ("pypdf", "pypdf", None),
    ("torch", "torch", None),
    ("transformers", "transformers", "4.46.3"),
    ("sentence_transformers", "sentence-transformers", "3.2.1"),
    ("sklearn", "scikit-learn", None),
    ("requests", "requests", None),
    ("numpy", "numpy", None),
    ("jieba", "jieba", None),
    ("rank_bm25", "rank-bm25", None),
)
CSS = """
.review-trace {line-height: 1.7; overflow-wrap: anywhere;}
.review-trace .trace-sentence, .review-trace .trace-evidence {
  border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px; margin: 12px 0;
}
.review-trace .trace-evidence {border-left: 5px solid #64748b;}
.review-trace .trace-evidence:target {
  background: #fff2b3 !important; color: #172033 !important;
  border: 3px solid #b77900; scroll-margin-top: 32px;
}
.review-trace .trace-warning {color: #9a4a00; font-weight: 600;}
.review-trace .trace-text {white-space: pre-wrap;}
.review-trace a {color: #1d4ed8; text-decoration: underline;}
.review-trace .trace-meta {font-size: 0.9em;}
.review-trace .trace-columns {
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px;
}
.review-trace .trace-review, .review-trace .trace-sources {
  min-width: 0; max-height: 70vh; overflow-y: auto; padding: 4px;
}
.review-trace .trace-sentence-link {display: block; padding: 6px 0;}
.review-trace a:focus-visible {outline: 3px solid #b77900; outline-offset: 3px;}
.review-trace .trace-evidence-group:target {scroll-margin-top: 16px;}
.review-trace:has(.trace-evidence-group:target, .trace-evidence:target)
  .trace-evidence-group:not(:target):not(:has(.trace-evidence:target)) {display: none;}
@media (max-width: 760px) {
  .review-trace .trace-columns {grid-template-columns: minmax(0, 1fr);}
  .review-trace .trace-review, .review-trace .trace-sources {max-height: none;}
}
@media print {
  .review-trace .trace-columns {display: block;}
  .review-trace .trace-review, .review-trace .trace-sources {max-height: none; overflow: visible;}
  .review-trace .trace-evidence-group {display: block !important;}
}
"""
DEMO_RESULT = {
    "query": "【演示】如何查看逐句证据？",
    "review": "这是一句用于展示溯源交互的示例。另一句示例暂时没有匹配证据。",
    "sentences": [
        {
            "sentence": "这是一句用于展示溯源交互的示例。",
            "evidences": [
                {
                    "page": 1,
                    "para_id": 1,
                    "source": "演示文献甲（手工构造）",
                    "text": "【手工构造的演示原文】点击证据标记后，这个原文片段会被定位并高亮。",
                    "nli_label": "entailment",
                    "nli_prob": 0.85,
                },
                {
                    "page": 2,
                    "para_id": 1,
                    "source": "演示文献乙（手工构造）",
                    "text": "【第二篇手工构造原文】段号同为 1，但文献名不同，不能合并为同一证据。",
                    "nli_label": "entailment",
                    "nli_prob": 0.78,
                },
            ],
        },
        {"sentence": "另一句示例暂时没有匹配证据。", "evidences": []},
    ],
}
_EXPORT_DIR: tempfile.TemporaryDirectory | None = None
_EXPORT_LOCK = threading.Lock()


def _text(value, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} 必须是文本。")
    if len(value) > 500_000:
        raise ValueError(f"{field} 过长（最多 500000 个字符）。")
    return value


def validate_result(value) -> dict:
    """Validate the current pipeline contract; never coerce malformed model output."""
    if not isinstance(value, dict):
        raise ValueError("结果必须是 JSON 对象。")
    result = {
        "query": _text(value.get("query"), "query"),
        "review": _text(value.get("review"), "review"),
        "sentences": [],
    }
    sentences = value.get("sentences")
    if not isinstance(sentences, list) or len(sentences) > 1000:
        raise ValueError("sentences 必须是数组，最多 1000 句。")
    for i, item in enumerate(sentences, 1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {i} 句必须是对象。")
        sentence = _text(item.get("sentence"), f"第 {i} 句 sentence")
        evidences = item.get("evidences")
        if not isinstance(evidences, list) or len(evidences) > 100:
            raise ValueError(f"第 {i} 句 evidences 必须是数组，最多 100 条。")
        checked = []
        for j, evidence in enumerate(evidences, 1):
            field = f"第 {i} 句第 {j} 条证据"
            if not isinstance(evidence, dict):
                raise ValueError(f"{field}必须是对象。")
            for key in ("page", "para_id"):
                if type(evidence.get(key)) is not int or evidence[key] < 1:
                    raise ValueError(f"{field} {key} 必须是正整数。")
            label = evidence.get("nli_label")
            if label not in ("entailment", "neutral", "contradiction"):
                raise ValueError(f"{field} nli_label 无效。")
            prob = evidence.get("nli_prob")
            if type(prob) not in (int, float) or not 0 <= prob <= 1 or not math.isfinite(prob):
                raise ValueError(f"{field} nli_prob 必须是 0 到 1 之间的有限数值。")
            checked.append(
                {
                    "source": _text(evidence.get("source", ""), f"{field} source"),
                    "page": evidence["page"],
                    "para_id": evidence["para_id"],
                    "text": _text(evidence.get("text"), f"{field} text"),
                    "nli_label": label,
                    "nli_prob": float(prob),
                }
            )
        result["sentences"].append({"sentence": sentence, "evidences": checked})
    return result


def _reject_constant(value):
    raise ValueError(f"JSON 不允许非有限数值 {value}。")


def _read_json(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise ValueError("请选择一个存在的 JSON 文件。")
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON 文件不能超过 5 MiB。")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("文件不是有效的 UTF-8 JSON。") from exc


def load_result(path: str | Path) -> tuple[dict, dict]:
    """Accept raw pipeline JSON or this app's export envelope; imports are unverified."""
    data = _read_json(path)
    metadata = {"source_name": "导入结果（来源未核验）", "is_demo": False}
    if isinstance(data, dict) and "schema_version" in data:
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("不支持的报告版本；当前支持 schema_version=1。")
        previous = data.get("metadata")
        if not isinstance(previous, dict) or type(previous.get("is_demo")) is not bool:
            raise ValueError("报告 metadata 或 is_demo 字段无效。")
        source = _text(previous.get("source_name"), "metadata.source_name")
        metadata = {
            "source_name": f"导入结果（来源未核验）：{source}",
            "is_demo": previous["is_demo"],
        }
        data = data.get("result")
    return validate_result(data), metadata


def _paragraph_checksum(paragraphs) -> str:
    """Member A ce04a5e's frozen-corpus serialization, not a PDF fingerprint."""
    return hashlib.sha256(
        json.dumps(paragraphs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def validate_frozen_corpus(payload) -> tuple[dict, str]:
    """Validate A's exported snapshot without importing an unmerged evaluation module."""
    if not isinstance(payload, dict):
        raise ValueError("请使用成员 A 工具导出的 corpus.json 对象。")
    paragraphs = payload.get("paragraphs")
    if not isinstance(paragraphs, list) or not paragraphs or len(paragraphs) > 100_000:
        raise ValueError("语料 paragraphs 必须是非空数组，最多 100000 段。")
    index, groups = {}, {}
    for paragraph in paragraphs:
        if not isinstance(paragraph, dict):
            raise ValueError("语料段落必须是对象。")
        source = _text(paragraph.get("source"), "语料 source")
        if not source.strip() or not _text(paragraph.get("text"), "语料 text").strip():
            raise ValueError("语料文献名和原文不能为空。")
        for field in ("page", "para_id"):
            if type(paragraph.get(field)) is not int or paragraph[field] < 1:
                raise ValueError(f"语料 {field} 必须是正整数。")
        key = (source, paragraph["para_id"])
        if key in index:
            raise ValueError("语料存在重复的 (source, para_id)。")
        index[key] = paragraph
        groups.setdefault(source, []).append(paragraph)
    checksum = _paragraph_checksum(paragraphs)
    manifest = payload.get("manifest")
    if (
        payload.get("parsed_paragraphs_sha256") != checksum
        or not isinstance(manifest, dict)
        or manifest.get("parsed_paragraphs_sha256") != checksum
        or type(manifest.get("paragraph_count")) is not int
        or manifest["paragraph_count"] != len(paragraphs)
    ):
        raise ValueError("语料快照或清单校验值/段落数不匹配，请重新导出并复核。")
    documents = manifest.get("documents")
    if not isinstance(documents, list) or len(documents) != len(groups):
        raise ValueError("语料文献清单不完整。")
    seen = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("文献清单条目必须是对象。")
        source = document.get("source")
        if not isinstance(source, str) or source not in groups or source in seen:
            raise ValueError("文献清单含未知或重复 source。")
        seen.add(source)
        if (
            type(document.get("paragraph_count")) is not int
            or document["paragraph_count"] != len(groups[source])
            or document.get("parsed_paragraphs_sha256") != _paragraph_checksum(groups[source])
        ):
            raise ValueError("文献清单的段落数或校验值不匹配。")
    return index, checksum


def check_corpus_evidence(corpus_path, result_path) -> str:
    """Read-only integration check; no relevance judging, scoring, or review promotion."""
    notice = (
        "仅核对所选两个文件，不核验原 PDF、论文真实性、人工复核或综述正确性；"
        "一致不代表相关性/NLI/正式评测通过。"
    )
    try:
        if not corpus_path or not result_path:
            raise ValueError("请同时选择冻结语料 corpus.json 和 pipeline/导出结果 JSON。")
        payload = _read_json(corpus_path)
        index, checksum = validate_frozen_corpus(payload)
        kind = _text(payload.get("dataset_type", "unverified"), "dataset_type")
        result, metadata = load_result(result_path)
        issues, total, matched, empty = [], 0, 0, 0
        for i, sentence in enumerate(result["sentences"], 1):
            if not sentence["evidences"]:
                empty += 1
            for j, evidence in enumerate(sentence["evidences"], 1):
                total += 1
                prefix = f"第 {i} 句第 {j} 条证据"
                original = index.get((evidence["source"], evidence["para_id"]))
                if not evidence["source"].strip():
                    issues.append(f"{prefix}：缺少文献名，无法核对。")
                elif original is None:
                    issues.append(f"{prefix}：(source, para_id) 不在冻结语料中。")
                else:
                    different = [k for k in ("page", "text") if evidence[k] != original[k]]
                    if different:
                        issues.append(f"{prefix}：{'、'.join(different)} 与冻结语料不一致。")
                    else:
                        matched += 1
        state = "发现不一致" if issues else ("证据字段一致" if total else "无证据可核对")
        lines = [
            f"核对结果：{state}。",
            f"语料类型声明（未核验）：{kind}",
            f"结果 SHA-256（规范化内容）：{_paragraph_checksum(result)}",
            f"语料 SHA-256：{checksum}",
            f"证据 {total} 条，一致 {matched} 条，不一致 {len(issues)} 条；无证据句 {empty} 句。",
            *issues[:100],
        ]
        if len(issues) > 100:
            lines.append(f"其余 {len(issues) - 100} 条问题未展开，请分批核查。")
        if metadata["is_demo"] or kind in ("synthetic", "demo", "test"):
            lines.append(DEMO_NOTICE)
        return "\n".join([*lines, notice])
    except (ValueError, OSError) as exc:
        return f"未完成核对：{exc}\n{notice}"


def _source_label(evidence: dict) -> str:
    return evidence["source"] if evidence["source"].strip() else UNKNOWN_SOURCE


def render_provenance(result: dict, *, anchor_prefix: str = "") -> str:
    """Select a sentence on the left to filter evidence on the right, without scripts.

    Fixed numeric anchors are independent of source names and paragraph IDs.
    Browsers without :has() still show every evidence with working anchor links.
    """
    result = validate_result(result)
    if not isinstance(anchor_prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", anchor_prefix):
        raise ValueError("证据锚点前缀无效。")
    escape = html.escape
    parts = ['<section class="review-trace" aria-label="逐句证据溯源">']
    if not result["sentences"]:
        parts.append('<p class="trace-warning">没有逐句结果，无法展示证据，请人工核查。</p>')
    parts.extend(
        [
            f'<p id="{anchor_prefix}trace-all">点击左侧句子，在右侧查看对应证据。'
            f'<a href="#{anchor_prefix}trace-all">查看全部证据</a>；'
            "旧版缺少文献名时显示“文献来源未知”。</p>",
            '<div class="trace-columns"><div class="trace-review" aria-label="综述逐句选择">',
            "<h3>综述 · 选择一句</h3>",
        ]
    )
    for i, item in enumerate(result["sentences"], 1):
        parts.append(f'<article class="trace-sentence"><strong>句 {i}</strong>')
        parts.append(
            f'<a class="trace-sentence-link trace-text" href="#{anchor_prefix}sentence-{i}">'
            f"{escape(item['sentence'])}</a>"
        )
        for j, evidence in enumerate(item["evidences"], 1):
            parts.append(
                f'<a href="#{anchor_prefix}ev-{i}-{j}">[证据{j}: '
                f"{escape(_source_label(evidence))} · "
                f"第{evidence['page']}页 第{evidence['para_id']}段]</a> "
            )
        if not item["evidences"]:
            parts.append('<p class="trace-warning">未匹配到证据，需人工核查。</p>')
        parts.append("</article>")
    parts.append(
        '</div><div class="trace-sources" aria-label="对应原文证据"><h3>原文证据 · 文献与位置</h3>'
    )
    for i, item in enumerate(result["sentences"], 1):
        parts.append(
            f'<section id="{anchor_prefix}sentence-{i}" class="trace-evidence-group" tabindex="0">'
            f"<h4>句 {i} 对应的原文片段</h4>"
        )
        if not item["evidences"]:
            parts.append('<p class="trace-warning">未匹配到证据，需人工核查。</p>')
        for j, evidence in enumerate(item["evidences"], 1):
            parts.append(
                f'<article id="{anchor_prefix}ev-{i}-{j}" class="trace-evidence" tabindex="0">'
                f"<strong>句 {i} · 证据 {j} · 文献：{escape(_source_label(evidence))} · "
                f"第{evidence['page']}页 第{evidence['para_id']}段</strong>"
                f'<p class="trace-meta">NLI 标签：{escape(evidence["nli_label"])}；'
                f"模型概率：{evidence['nli_prob']:.3f}</p>"
                f'<p class="trace-text">{escape(evidence["text"])}</p></article>'
            )
        parts.append("</section>")
    parts.append("</div></div></section>")
    return "\n".join(parts)


def _markdown(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", html.escape(text)).replace("\n", "<br>")


def _csv_cell(text: str) -> str:
    # Neutralize spreadsheet formulas, including leading whitespace/control characters.
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r")):
        return "'" + text
    return text


def export_report(
    result: dict,
    output_dir: str | Path | None = None,
    *,
    source_name: str = "导入结果（来源未核验）",
    is_demo: bool = False,
) -> list[str]:
    """Create unique per-request reports; user strings never become paths or HTML."""
    global _EXPORT_DIR
    result = validate_result(result)
    source_name = _text(source_name, "source_name")
    if type(is_demo) is not bool:
        raise ValueError("is_demo 必须是布尔值。")
    if output_dir is None:
        with _EXPORT_LOCK:
            if _EXPORT_DIR is None:
                _EXPORT_DIR = tempfile.TemporaryDirectory(prefix="trusted-review-ui-")
            output_dir = _EXPORT_DIR.name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="report-", dir=output_dir))
    metadata = {
        "source_name": source_name,
        "is_demo": is_demo,
        "exported_at": datetime.now(UTC).isoformat(),
        "notice": NOTICE,
        "evaluation_status": "未进行正式评测；等待队长确认指标公式和人工标注。",
    }
    warning = f"{DEMO_NOTICE}\n{NOTICE}" if is_demo else NOTICE
    envelope = {"schema_version": 1, "metadata": metadata, "result": result}
    json_file = folder / "result.json"
    json_file.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2, allow_nan=False), "utf-8"
    )
    md = [
        "# 科研综述溯源报告",
        "",
        _markdown(warning),
        "",
        f"来源：{_markdown(source_name)}",
        "",
        "## 用户问题",
        _markdown(result["query"]),
        "",
        "## 综述",
        _markdown(result["review"]),
        "",
        "## 逐句证据对照",
        "",
        "| 句号 | 生成句 | 文献名 | 页码 / 段号 | 原文 | NLI 标签 / 模型概率 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    csv_file = folder / "evidence.csv"
    with csv_file.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "句号",
                "生成句",
                "页码",
                "段号",
                "原文",
                "NLI标签",
                "模型概率",
                "来源",
                "说明",
                "文献名",
            ]
        )
        for i, item in enumerate(result["sentences"], 1):
            evidences = item["evidences"] or [None]
            for evidence in evidences:
                if evidence is None:
                    cells = [i, item["sentence"], "", "", "未匹配到证据，需人工核查", "", ""]
                    row = (
                        f"| {i} | {_markdown(item['sentence'])} | — | — | "
                        "未匹配到证据，需人工核查 | — |"
                    )
                else:
                    cells = [
                        i,
                        item["sentence"],
                        evidence["page"],
                        evidence["para_id"],
                        evidence["text"],
                        evidence["nli_label"],
                        evidence["nli_prob"],
                    ]
                    row = (
                        f"| {i} | {_markdown(item['sentence'])} | "
                        f"{_markdown(_source_label(evidence))} | "
                        f"{evidence['page']} / {evidence['para_id']} | "
                        f"{_markdown(evidence['text'])} | {evidence['nli_label']} / "
                        f"{evidence['nli_prob']:.3f} |"
                    )
                writer.writerow(
                    [
                        _csv_cell(str(cell))
                        for cell in [
                            *cells,
                            source_name,
                            warning,
                            _source_label(evidence) if evidence else "",
                        ]
                    ]
                )
                md.append(row)
    if not result["sentences"]:
        md.append("\n没有逐句结果，无法展示证据，请人工核查。")
    md.extend(["", "## 评测状态", metadata["evaluation_status"], ""])
    md_file = folder / "report.md"
    md_file.write_text("\n".join(md), "utf-8")
    html_file = folder / "report.html"
    html_file.write_text(
        '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>科研综述溯源报告</title><style>"
        + CSS
        + "body {max-width: 1000px; margin: 32px auto; padding: 16px; font-family: sans-serif;}"
        "</style></head><body><h1>科研综述溯源报告</h1>"
        f"<p>{html.escape(warning)}</p><p>来源：{html.escape(source_name)}</p>"
        f'<h2>用户问题</h2><p class="trace-text">{html.escape(result["query"])}</p>'
        f'<h2>综述</h2><p style="white-space:pre-wrap">{html.escape(result["review"])}</p>'
        + render_provenance(result)
        + f"<h2>评测状态</h2><p>{html.escape(metadata['evaluation_status'])}</p></body></html>",
        "utf-8",
    )
    return [str(path) for path in (md_file, csv_file, json_file, html_file)]


def environment_issues() -> list[str]:
    """Inspect prerequisites without importing heavy model modules or downloading data."""
    issues = []
    if sys.version_info[:2] != (3, 12):
        issues.append("项目要求 Python 3.12；请使用兼容的核心 Python 环境。")
    for name, distribution, required_version in CORE_DEPENDENCIES:
        if importlib.util.find_spec(name) is None:
            issues.append(f"核心环境缺少 {name}；请使用已配置模型依赖的 Python。")
        elif required_version:
            try:
                installed = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                issues.append(f"无法确认 {distribution} 版本；项目锁定 {required_version}。")
            else:
                if installed != required_version:
                    issues.append(
                        f"{distribution} 版本为 {installed}；项目锁定 {required_version}，"
                        "请使用兼容的核心环境，不要自动升级模型依赖。"
                    )
    if not (ROOT / "models" / "bge-small-zh-v1.5").is_dir():
        issues.append("缺少 models/bge-small-zh-v1.5 本地 BGE 权重。")
    try:
        module = ast.parse((SRC / "match.py").read_text(encoding="utf-8"))
        nli_path = next(
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_NLI_PATH" for target in node.targets
            )
        )
        if not isinstance(nli_path, str) or not (SRC / nli_path).is_dir():
            issues.append("match.py 的 _NLI_PATH 不存在；请队长配置本地 NLI 权重路径。")
    except (OSError, SyntaxError, ValueError, StopIteration):
        issues.append("无法确认 match.py 的 NLI 路径，请由核心链路维护人检查。")
    try:
        with socket.create_connection(("127.0.0.1", 8080), timeout=1):
            pass
    except OSError:
        issues.append("127.0.0.1:8080 不可连接；请启动本地 llama-server。")
    return issues


def _offline_environment() -> dict:
    return {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUTF8": "1",
    }


def environment_report() -> dict:
    """Describe the current interpreter, never a different configured backend."""
    return {
        "schema_version": 1,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "issues": environment_issues(),
        "notice": CHECK_NOTICE,
    }


def check_backend_environment(backend_python=None, *, timeout=15) -> dict:
    """Probe the same Python and cwd used by generation, without importing model code."""
    try:
        completed = subprocess.run(
            [
                backend_python or sys.executable,
                str(Path(__file__).resolve()),
                "--check",
                "--json",
            ],
            cwd=SRC,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=_offline_environment(),
        )
    except subprocess.TimeoutExpired as exc:
        raise ValueError(f"环境自检超过 {timeout} 秒，已停止；请检查核心 Python。") from exc
    except OSError as exc:
        raise ValueError("无法启动核心 Python；请检查 --backend-python 路径。") from exc
    try:
        report = json.loads(completed.stdout, parse_constant=_reject_constant)
        if (
            not isinstance(report, dict)
            or type(report.get("schema_version")) is not int
            or report["schema_version"] != 1
            or not isinstance(report.get("issues"), list)
            or not all(isinstance(issue, str) for issue in report["issues"])
            or not isinstance(report.get("python_executable"), str)
            or not report["python_executable"]
            or not isinstance(report.get("python_version"), str)
            or not report["python_version"]
            or completed.returncode != int(bool(report["issues"]))
        ):
            raise ValueError("自检响应格式或退出码无效。")
    except (ValueError, RecursionError) as exc:
        reason = completed.stderr.strip()[-1000:] or "核心进程没有返回有效的自检报告。"
        raise ValueError(f"环境自检未完成：{reason}") from exc
    report["notice"] = CHECK_NOTICE
    return report


def format_environment_report(report: dict) -> str:
    status = "基础前置检查通过。" if not report["issues"] else "基础前置检查未通过："
    lines = [
        f"检查的 Python：{report['python_executable']}",
        f"Python 版本：{report['python_version']}",
        status,
        *(f"- {issue}" for issue in report["issues"]),
        CHECK_NOTICE,
    ]
    return "\n".join(lines)


def check_environment(backend_python=None) -> str:
    try:
        return format_environment_report(check_backend_environment(backend_python))
    except ValueError as exc:
        return f"未完成：{exc}\n{CHECK_NOTICE}"


def _normalize_pdf_paths(pdf_paths) -> list:
    if isinstance(pdf_paths, (str, Path)):
        return [pdf_paths] if pdf_paths else []
    if pdf_paths is None:
        return []
    if not isinstance(pdf_paths, (list, tuple)):
        raise ValueError("请选择一个或多个 PDF 文件。")
    return list(pdf_paths)


def _validate_request(query, pdf_paths, top_k, nli_threshold) -> tuple[str, list[Path]]:
    query = _text(query, "问题").strip()
    if not query:
        raise ValueError("请输入问题。")
    if len(query) > 4000:
        raise ValueError("问题不能超过 4000 个字符。")
    paths = _normalize_pdf_paths(pdf_paths)
    if not paths:
        raise ValueError("请先上传文本型 PDF。")
    if len(paths) > MAX_PDF_FILES:
        raise ValueError(f"一次最多上传 {MAX_PDF_FILES} 篇 PDF。")
    pdfs, sources, total_size = [], set(), 0
    for path in paths:
        if not isinstance(path, (str, Path)) or not path:
            raise ValueError("请选择有效的 PDF 文件路径。")
        pdf = Path(path).resolve()
        if not pdf.is_file() or pdf.suffix.lower() != ".pdf":
            raise ValueError("请选择有效的 PDF 文件。")
        stem = unicodedata.normalize("NFC", pdf.stem).casefold()
        if not stem.strip() or stem in sources:
            raise ValueError("PDF 文献名（不含扩展名）不能为空或重复；请重命名后重新上传。")
        sources.add(stem)
        size = pdf.stat().st_size
        if size > MAX_PDF_BYTES:
            raise ValueError("每篇 PDF 不能超过 50 MiB。")
        total_size += size
        if total_size > MAX_TOTAL_PDF_BYTES:
            raise ValueError("PDF 总大小不能超过 200 MiB。")
        with pdf.open("rb") as stream:
            if b"%PDF-" not in stream.read(1024):
                raise ValueError("文件没有有效的 PDF 标识；不支持扫描件 OCR。")
        pdfs.append(pdf)
    if type(top_k) not in (int, float) or not 1 <= top_k <= 20 or not float(top_k).is_integer():
        raise ValueError("top_k 必须是 1 到 20 的整数。")
    if (
        type(nli_threshold) not in (int, float)
        or not 0 <= nli_threshold <= 1
        or not math.isfinite(nli_threshold)
    ):
        raise ValueError("NLI 阈值必须在 0 到 1 之间。")
    return query, pdfs


def run_pipeline(
    query,
    pdf_paths,
    top_k=3,
    nli_threshold=0.7,
    *,
    timeout=600,
    backend_python=None,
) -> dict:
    """Run the owner's unchanged pipeline from src, with a deadline and offline models."""
    query, pdfs = _validate_request(query, pdf_paths, top_k, nli_threshold)
    with tempfile.TemporaryDirectory(prefix="trusted-review-job-") as directory:
        request_file, result_file = (
            Path(directory) / "request.json",
            Path(directory) / "result.json",
        )
        request_file.write_text(
            json.dumps(
                {
                    "query": query,
                    "pdf_paths": [str(pdf) for pdf in pdfs],
                    "top_k": int(top_k),
                    "nli_threshold": nli_threshold,
                },
                ensure_ascii=False,
            ),
            "utf-8",
        )
        try:
            completed = subprocess.run(
                [
                    backend_python or sys.executable,
                    str(Path(__file__).resolve()),
                    "--pipeline-job",
                    str(request_file),
                    str(result_file),
                ],
                cwd=SRC,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
                env=_offline_environment(),
            )
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                f"生成超过 {timeout} 秒，已停止本次任务；请检查本地模型服务。"
            ) from exc
        except OSError as exc:
            raise ValueError("无法启动核心 Python；请检查 --backend-python 路径。") from exc
        if completed.returncode:
            reason = completed.stderr.strip()[-3000:] or "核心进程未返回结果。"
            raise ValueError(f"生成未完成：{reason}")
        return load_result(result_file)[0]


def _pipeline_worker(request_file: str, result_file: str) -> int:
    try:
        request = _read_json(request_file)
        issues = environment_issues()
        if issues:
            raise ValueError("\n".join(issues))
        # Existing src-level imports need this process's cwd and sys.path to be src.
        from pipeline import run_review

        result = validate_result(run_review(**request))
        Path(result_file).write_text(
            json.dumps(result, ensure_ascii=False, allow_nan=False), "utf-8"
        )
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def _display(result: dict, source_name: str, is_demo: bool = False) -> tuple:
    result = validate_result(result)
    sentence_count = len(result["sentences"])
    matched_count = sum(bool(item["evidences"]) for item in result["sentences"])
    status = (
        f"{source_name}\n本次问题：{result['query']}\n"
        f"共 {sentence_count} 句，其中 {matched_count} 句返回证据，"
        f"{sentence_count - matched_count} 句未返回证据。这里只统计结果数量，不是正式评测。"
    )
    if is_demo:
        status = DEMO_NOTICE + "\n" + status
    files = export_report(result, source_name=source_name, is_demo=is_demo)
    rendered = render_provenance(result, anchor_prefix=f"run-{secrets.token_hex(8)}-")
    return status, result["review"], rendered, files


def _failed(exc: Exception) -> tuple:
    # Clear all old outputs on failure, so an older report cannot look like this run's result.
    return f"未完成：{exc}", "", "", None


def import_result(path):
    try:
        if not path:
            raise ValueError("请先选择 pipeline 结果 JSON。")
        result, metadata = load_result(path)
        return _display(result, **metadata)
    except (ValueError, OSError) as exc:
        return _failed(exc)


def show_demo():
    return _display(DEMO_RESULT, "内置手工演示样例", is_demo=True)


def generate_review(query, pdf_paths, top_k, nli_threshold, *, backend_python=None):
    yield "正在运行本地 pipeline，最长等待 10 分钟……", "", "", None
    try:
        result = run_pipeline(query, pdf_paths, top_k, nli_threshold, backend_python=backend_python)
        names = [Path(path).name for path in _normalize_pdf_paths(pdf_paths)]
        yield _display(result, f"本地生成 · {len(names)} 篇 PDF：" + "、".join(names))
    except (ValueError, OSError) as exc:
        yield _failed(exc)


def build_app(backend_python=None):
    import gradio as gr

    with gr.Blocks(
        title="可信可溯源科研综述辅助系统",
        css=CSS,
        analytics_enabled=False,
        delete_cache=(3600, 86400),
    ) as demo:
        gr.Markdown("# 科研综述 · 逐句溯源\n本地生成、查看证据、人工核查、导出报告。")
        gr.Markdown(f"**使用边界：{NOTICE}**")
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("从文本型 PDF 生成"):
                    pdf = gr.File(
                        label="文本型 PDF 文献库（1–10 篇，每篇 ≤50 MiB，总计 ≤200 MiB）",
                        file_types=[".pdf"],
                        file_count="multiple",
                    )
                    gr.Markdown("中期演示建议 3–5 篇。文件名（不含扩展名）必须互不重复。")
                    query = gr.Textbox(
                        label="研究问题",
                        lines=3,
                        placeholder="这些论文采用了哪些方法，有什么共同点与差异？",
                    )
                    with gr.Accordion("检索与匹配参数", open=False):
                        top_k = gr.Slider(1, 20, value=3, step=1, label="每句候选证据数量 top_k")
                        threshold = gr.Slider(0, 1, value=0.7, step=0.05, label="NLI 匹配阈值")
                    submit = gr.Button("生成综述并匹配证据", variant="primary")
                    gr.Markdown(
                        "需要已配置的本地 BGE / NLI / llama-server。"
                        "缺少模型时使用下方离线导入或演示；不会自动下载模型。"
                    )
                    with gr.Accordion("真实生成环境自检（不运行模型）", open=False):
                        check_button = gr.Button("检查核心 Python 与本地模型前置条件")
                        diagnostics = gr.Textbox(
                            label="自检结果（不代表真实联调通过）", interactive=False, lines=8
                        )
                with gr.Tab("导入已有结果"):
                    imported = gr.File(
                        label="pipeline 结果 JSON（最大 5 MiB）", file_types=[".json"]
                    )
                    import_button = gr.Button("导入并展示")
                with gr.Tab("联调语料核对"):
                    gr.Markdown(
                        "分别上传成员 A 导出的冻结语料和待核对结果（各 ≤5 MiB）。"
                        "按文献名+段号检查页码及原文，不加载模型、不改标签、不计算效果分数。"
                    )
                    corpus_file = gr.File(label="冻结语料 corpus.json", file_types=[".json"])
                    check_result_file = gr.File(label="待核对结果 JSON", file_types=[".json"])
                    corpus_button = gr.Button("核对证据与冻结语料")
                    corpus_status = gr.Textbox(
                        label="两个所选文件的核对结果", lines=10, interactive=False
                    )
                demo_button = gr.Button("加载演示样例（非真实结果）")
                clear_button = gr.Button("清空结果")
                gr.Markdown(
                    "检索 P/R/MRR 由成员 A 的工具评测，需真实语料和复核标签；"
                    "综述/NLI 效果另行验收。"
                    "点击证据高亮的是返回的原文片段，不是 PDF 原页坐标。"
                )
            with gr.Column(scale=2):
                status = gr.Textbox(label="运行状态与来源", interactive=False, lines=3)
                review = gr.Textbox(label="综述全文", interactive=False, lines=5)
                provenance = gr.HTML(label="逐句证据溯源", padding=True)
                files = gr.File(
                    label="下载报告：Markdown / CSV / JSON / HTML", file_count="multiple"
                )
        outputs = [status, review, provenance, files]
        shared = {"outputs": outputs, "concurrency_id": "review-actions", "concurrency_limit": 1}

        def generate(query, pdf, top_k, threshold):
            yield from generate_review(query, pdf, top_k, threshold, backend_python=backend_python)

        submit.click(generate, [query, pdf, top_k, threshold], api_name="generate_review", **shared)
        import_button.click(import_result, [imported], api_name="import_result", **shared)
        demo_button.click(show_demo, api_name="show_demo", **shared)
        clear_button.click(
            lambda: ("已清空结果。", "", "", None), api_name="clear_result", **shared
        )
        corpus_button.click(
            check_corpus_evidence,
            [corpus_file, check_result_file],
            [corpus_status],
            api_name="check_corpus_evidence",
            concurrency_id="review-actions",
            concurrency_limit=1,
        )
        for input_file in (corpus_file, check_result_file):
            input_file.change(
                lambda: "输入已变化，请重新核对。",
                outputs=[corpus_status],
                concurrency_id="review-actions",
                concurrency_limit=1,
            )
        check_button.click(
            lambda: check_environment(backend_python),
            outputs=[diagnostics],
            api_name="check_environment",
            concurrency_id="review-actions",
            concurrency_limit=1,
        )
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--backend-python", help="已有核心模型虚拟环境的 Python 路径")
    parser.add_argument("--check", action="store_true", help="只检查所选核心 Python 的生成前置条件")
    parser.add_argument("--json", action="store_true", help="与 --check 一起输出机器可读自检报告")
    parser.add_argument(
        "--pipeline-job", nargs=2, metavar=("REQUEST", "RESULT"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.json and not args.check:
        parser.error("--json 仅可与 --check 一起使用。")
    if args.pipeline_job:
        return _pipeline_worker(*args.pipeline_job)
    if args.check:
        try:
            report = (
                check_backend_environment(args.backend_python)
                if args.backend_python
                else environment_report()
            )
        except ValueError as exc:
            if args.json:
                print(json.dumps({"error": str(exc), "notice": CHECK_NOTICE}, ensure_ascii=False))
            else:
                print(f"未完成：{exc}\n{CHECK_NOTICE}")
            return 2
        print(
            json.dumps(report, ensure_ascii=False, indent=2)
            if args.json
            else format_environment_report(report)
        )
        return int(bool(report["issues"]))
    build_app(args.backend_python).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        show_error=False,
        inbrowser=False,
        max_file_size="50mb",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
