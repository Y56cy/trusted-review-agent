"""Member B's local UI: consume the existing pipeline without changing its contract.

Gradio and Word export dependencies are loaded only when used. Model code runs
in a separate, time-bounded process.
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
NOTICE = "句级匹配仅提供证据支持与人工核查依据，不证明事实正确。 NLI 概率不是事实准确率。"
MATCH_LABELS = {"entailment": "支持", "neutral": "中立", "contradiction": "矛盾"}
CORPUS_STATUS = {
    True: ("corpus-verified", "✓ 已通过语料一致性校验"),
    False: ("corpus-mismatch", "⚠ 与冻结语料不一致"),
    None: ("corpus-unchecked", "未校验"),
}
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
.review-trace .evidence-badges {display:flex; flex-wrap:wrap; gap:6px; margin:8px 0 14px;}
.review-trace .evidence-badge {border-radius:5px; padding:2px 8px; font-size:13px;
  color:#314a70; background:#edf3ff; overflow-wrap:anywhere;}
.review-trace .evidence-source {display:inline-block; font-weight:600;}
.review-trace .corpus-verified {color:#126044; background:#e7f6ed;}
.review-trace .corpus-mismatch {color:#8a4708; background:#fff0df;}
.review-trace .corpus-unchecked {color:#48576b; background:#edf0f4;}
.review-trace .trace-columns {
  display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px;
}
.review-trace .trace-review, .review-trace .trace-sources {
  min-width: 0; max-height: 70vh; overflow-y: auto; padding: 4px;
}
.review-trace .trace-sentence-link {display: block; padding: 6px 0;}
.review-trace .trace-citation {display: block; padding: 4px 0; overflow-wrap: anywhere;}
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
# App shell styles stay out of standalone report exports.
UI_CSS = """
.gradio-container {width:100% !important; max-width:none !important; margin:0 !important;
  padding:0 clamp(12px, 2.5vw, 40px) !important;
  font-family: "Inter", "Noto Sans SC", "Microsoft YaHei", sans-serif !important;}
.gradio-container > main {padding:12px 0 !important;}
body, .gradio-container {background: #f7f8fa !important; color: #202b3b;}
footer {display:none !important;}
.workspace-header {padding:14px 0; border-bottom:1px solid #e1e5ec;}
.workspace-header h1 {font-size:24px; font-weight:700;
  margin:0; color:#17243c;}
#app-header {padding:0; border:0; background:transparent;}
#workspace-layout {--workspace-height:max(400px, calc(100dvh - 114px));
  display:grid; grid-template-columns:clamp(268px, 22vw, 328px) minmax(0, 1fr);
  align-items:start; gap:clamp(14px, 1.6vw, 24px);}
#input-panel {min-width:0 !important; padding:18px; gap:14px;
  background:#fff; border:1px solid #dde3ec; border-radius:10px;}
#input-panel button, #output-panel button {box-shadow: none; border-radius:6px;}
#input-panel .block, #output-panel .block {box-shadow: none;}
#utility-actions {gap:8px; flex-grow:0 !important;}
#utility-actions button {width:100%; min-width:0 !important;}
#demo-action, #clear-action {background:transparent; color:#506078; border:1px solid #dde3ec;}
#demo-action {font-size:14px;}
#advanced-panel {padding:8px 10px;}
#clear-action {white-space:nowrap; min-width:84px !important; font-size:13px;}
#run-status label {background:transparent !important; border:0 !important;}
#output-panel {--reader-pane-height:var(--workspace-height);
  padding:0; min-width:0 !important; gap:8px;}
#output-panel:has(#provenance .reader) {
  --reader-pane-height:max(400px, calc(var(--workspace-height) - 100px));
}
#reading-toolbar {display:grid; grid-template-columns:minmax(0, 1fr) auto;
  gap:8px; align-items:center;}
#reading-toolbar > :has(> #run-status) {min-width:0 !important;}
#result-actions {display:grid; grid-template-columns:140px 140px 84px; align-items:start; gap:8px;}
#result-actions:has(.label-wrap.open) {grid-column:1 / -1;
  grid-template-columns:repeat(2, minmax(0, 1fr)) 84px;}
#workspace-layout:not(:has(#provenance .reader)) :is(#result-actions, #report-panel),
#provenance:not(:has(.reader)),
#output-panel:has(#provenance .reader) #empty-reader {display:none !important;}
#reading-toolbar:has(#run-status textarea:placeholder-shown) {
  display:none;
}
#run-status {border:0; padding:0; background:transparent;}
#run-status .input-container {background:transparent !important;}
#run-status textarea {border:0; padding:8px 10px; min-height:40px; font-size:13px; color:#506078;
  line-height:1.65; box-shadow:none; background:#f7f8fa !important;}
#export-panel, #original-review-panel {border:1px solid #dde3ec; border-radius:6px;
  background:#fff; width:100%; min-width:0 !important;}
#provenance, #empty-reader {padding:0; border:0; background:transparent;}
.reader {background:#fff; border:1px solid #e1e6ed; border-radius:10px; overflow:hidden;}
.reader .reader-bar {display:flex; align-items:center; justify-content:space-between;
  padding:15px 24px; border-bottom:1px solid #e6eaf0; font-size:11px; color:#778296;}
.reader .reader-bar strong {color:#3d4f68; font-weight:500;}
.reader .trace-columns {grid-template-columns:minmax(0, 2fr) minmax(0, 3fr);
  gap:0; min-height:var(--reader-pane-height);}
.reader .trace-review, .reader .trace-sources {padding:clamp(16px, 2vw, 28px);
  max-height:var(--reader-pane-height);}
.reader .trace-review {font-size:16px; line-height:1.7;}
.reader .trace-sources {border-left:1px solid #e6eaf0; background:#fcfdff;}
.reader h2.reader-title {font-size:26px; letter-spacing:-.5px; line-height:1.5;
  font-weight:650; margin:0 0 14px; color:#17243c;}
.reader .reader-eyebrow {color:#3469bc; font-size:11px; letter-spacing:1px; margin:0 0 12px;}
.reader .reader-summary {font-size:12px; color:#778296; border-bottom:1px solid #e8ecf2;
  margin:0 0 26px; padding:0 0 22px;}
.reader .trace-sentence {border:0; border-radius:5px; padding:10px 12px;
  margin:0 -12px 13px; background:transparent; position:relative;}
.reader .trace-sentence > strong {font-size:10px; color:#9aa5b5; display:block; margin-bottom:3px;}
.reader .trace-sentence-link {font-size:16px; line-height:1.7; color:#26364c;
  text-decoration:none; padding:0;}
.reader .trace-sentence-link:hover {color:#205ac1;}
.reader .trace-sentence:has(a:hover), .reader .trace-sentence:focus-within {background:#f0f5ff;}
.reader .trace-citation {display:inline-block; color:#3768b3; background:#edf3ff;
  padding:2px 7px; margin:8px 5px 0 0; border-radius:4px; font-size:11px; text-decoration:none;}
.reader .trace-citation:hover {background:#dce8ff;}
.reader .trace-sources h3 {font-size:15px; color:#23334c; margin:0 0 20px; font-weight:600;}
.reader .trace-evidence-group > h4 {font-size:11px; font-weight:400;
  color:#8490a0; margin:0 0 18px;}
.reader .trace-evidence {border:0; border-top:1px solid #e3e8ef; border-radius:0;
  padding:20px 0; margin:0; background:transparent;}
.reader .trace-evidence:target {background:#f5f8ff !important; color:#26364c !important;
  border:0; border-top:2px solid #4275c8; padding:18px 10px; scroll-margin-top:20px;}
.reader .evidence-source {font-size:15px; line-height:1.7; color:#26364c;}
.reader .trace-text {overflow-wrap:anywhere;}
.reader .evidence-excerpt {font-size:14px; line-height:1.95; border-left:2px solid #c8d5ea;
  padding-left:14px; margin:0 0 16px; color:#40506a;}
.reader details {font-size:11px; color:#7a8799;}
.reader summary {cursor:pointer; padding:5px 0;}
.reader .trace-warning {font-size:11px; line-height:1.7; color:#9a681f; font-weight:400;}
.reader .reader-footnote {border-top:1px solid #e8ecf2; padding-top:20px;
  margin-top:30px; color:#8792a2; font-size:11px; line-height:1.8;}
.reader:not(:has(:target)) .trace-evidence-group:not(:first-of-type) {display:none;}
.reader:has(.reader-all:target) .trace-evidence-group {display:block;}
.reader-empty h2 {font-size:16px; font-weight:600; color:#23334c; margin:0 0 14px;}
.reader-empty p {font-size:14px; line-height:1.7; color:#506078; margin:0;}
#review-full textarea {font-size:16px; line-height:1.7;}
@media (min-width: 981px) {
 #input-panel {min-height:var(--workspace-height);}
 #input-panel > :has(> #research-question) {flex:1 1 auto !important;}
 #research-question, #research-question > label, #research-question .input-container {
   display:flex; flex-direction:column; flex:1 1 auto; min-height:0;
 }
 #research-question textarea {flex:1 1 auto; min-height:100px;}
 #pdf-upload button[aria-dropeffect="copy"]:not(.icon-mode) {
   height:clamp(128px, 23dvh, 260px) !important;
 }
}
@media (max-width: 980px) {
 #workspace-layout {grid-template-columns:minmax(0, 1fr);}
 #output-panel, #output-panel:has(#provenance .reader) {--reader-pane-height:auto;}
}
@media (max-width: 760px) {
 .workspace-header h1 {font-size:20px;}
 #reading-toolbar {grid-template-columns:minmax(0, 1fr);}
 #result-actions, #result-actions:has(.label-wrap.open) {
   grid-template-columns:repeat(2, minmax(0, 1fr)) 84px;
 }
 #result-actions > :is(#export-panel, #original-review-panel):has(.label-wrap.open) {
   grid-column:1 / -1;
 }
 .reader .trace-columns {grid-template-columns:minmax(0, 1fr);}
 .reader .trace-review, .reader .trace-sources {max-height:none; padding:16px;}
 .reader .trace-sources {border-left:0; border-top:1px solid #e6eaf0;}
 .reader-empty .trace-sources {min-height:80px;}
 .reader h2.reader-title {font-size:22px;}
 .reader .reader-bar {padding:12px 18px; gap:12px;}
}
"""
EMPTY_WORKSPACE = (
    '<section class="review-trace reader reader-empty" aria-label="综述与引用证据">'
    '<div class="trace-columns"><div class="trace-review"><h2>综述</h2>'
    "<p>添加文献并输入研究问题，或加载演示样例。</p></div>"
    '<div class="trace-sources"><h2>引用证据</h2></div></div></section>'
)

DEMO_RESULT = {
    "query": "【演示】延长开放时间对图书馆使用有什么影响？",
    "review": "延长开放时间后，图书馆晚间到访增加。现有材料还无法判断周末到访是否也有变化。",
    "sentences": [
        {
            "sentence": "延长开放时间后，图书馆晚间到访增加。",
            "evidences": [
                {
                    "page": 1,
                    "para_id": 1,
                    "source": "演示文献甲 · 开放时间调整记录",
                    "text": "闭馆时间延后一小时后，工作日晚间的入馆人数较调整前有所增加。",
                    "nli_label": "entailment",
                    "nli_prob": 0.85,
                },
                {
                    "page": 2,
                    "para_id": 1,
                    "source": "演示文献乙 · 读者使用情况调查",
                    "text": "部分受访读者表示，闭馆时间延后后，他们更常在晚间到馆阅读或自习。",
                    "nli_label": "entailment",
                    "nli_prob": 0.78,
                },
            ],
        },
        {"sentence": "现有材料还无法判断周末到访是否也有变化。", "evidences": []},
    ],
}
_EXPORT_DIR: tempfile.TemporaryDirectory | None = None
_EXPORT_LOCK = threading.Lock()


class ReviewBackendError(ValueError):
    """The local generation process could not complete."""


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
            canonical = {
                "source": _text(evidence.get("source", ""), f"{field} source"),
                "page": evidence["page"],
                "para_id": evidence["para_id"],
                "text": _text(evidence.get("text"), f"{field} text"),
                "nli_label": label,
                "nli_prob": float(prob),
            }
            if "corpus_verified" in evidence:
                verified = evidence["corpus_verified"]
                if verified is not None and type(verified) is not bool:
                    raise ValueError(f"{field} corpus_verified 必须是布尔值或 null。")
                canonical["corpus_verified"] = verified
            checked.append(canonical)
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


def load_frozen_corpus(path) -> tuple[dict | None, str]:
    """Replace this session's corpus with a validated index or clear its state."""
    if not path:
        return None, "未加载冻结语料。"
    try:
        index, _ = validate_frozen_corpus(_read_json(path))
    except (ValueError, OSError) as exc:
        return None, f"冻结语料未加载：{exc}"
    return index, f"已加载冻结语料（{len(index)} 段）。后续生成会自动核对证据。"


def check_corpus_evidence(result: dict, corpus_index: dict | None) -> dict:
    """Return a copy annotated by exact document, paragraph, page and text matches."""
    checked = validate_result(result)
    for sentence in checked["sentences"]:
        for evidence in sentence["evidences"]:
            verified = None
            if corpus_index is not None:
                original = corpus_index.get((evidence["source"], evidence["para_id"]))
                verified = original is not None and all(
                    evidence[key] == original[key] for key in ("page", "text")
                )
            evidence["corpus_verified"] = verified
    return checked


def _source_label(evidence: dict) -> str:
    return evidence["source"] if evidence["source"].strip() else UNKNOWN_SOURCE


def render_provenance(result: dict, *, anchor_prefix: str = "", workspace: bool = False) -> str:
    """Select a sentence on the left to filter evidence on the right, without scripts.

    Fixed numeric anchors are independent of source names and paragraph IDs.
    Browsers without :has() still show every evidence with working anchor links.
    """
    result = validate_result(result)
    if not isinstance(anchor_prefix, str) or not re.fullmatch(r"[A-Za-z0-9_-]*", anchor_prefix):
        raise ValueError("证据锚点前缀无效。")
    escape = html.escape
    parts = [
        '<section class="review-trace reader" aria-label="综述与引用证据">'
        if workspace
        else '<section class="review-trace" aria-label="逐句证据溯源">'
    ]
    if workspace:
        parts.append(
            f'<div class="reader-bar"><strong>阅读与核查</strong><a class="reader-all" '
            f'id="{anchor_prefix}trace-all" href="#{anchor_prefix}trace-all">全部证据</a></div>'
        )
    else:
        parts.append(
            f'<p id="{anchor_prefix}trace-all">点击左侧句子，在右侧查看对应证据。'
            f'<a href="#{anchor_prefix}trace-all">查看全部证据</a>；'
            "旧版缺少文献名时显示“文献来源未知”。</p>"
        )
    parts.append('<div class="trace-columns"><div class="trace-review" aria-label="综述逐句选择">')
    if workspace:
        parts.append(
            '<p class="reader-eyebrow">文献综述</p>'
            f'<h2 class="reader-title">{escape(result["query"])}</h2>'
            f'<p class="reader-summary">{len(result["sentences"])} 句综述 · 点击句子查看证据</p>'
        )
    else:
        parts.append("<h3>综述 · 选择一句</h3>")
    if not result["sentences"]:
        parts.append('<p class="trace-warning">没有逐句结果，无法展示证据，请人工核查。</p>')
        if workspace:
            parts.append(f'<p class="trace-text">{escape(result["review"])}</p>')
    for i, item in enumerate(result["sentences"], 1):
        parts.append(f'<article class="trace-sentence"><strong>句 {i}</strong>')
        parts.append(
            f'<a class="trace-sentence-link trace-text" href="#{anchor_prefix}sentence-{i}">'
            f"{escape(item['sentence'])}</a>"
        )
        for j, evidence in enumerate(item["evidences"], 1):
            label = (
                f"证据 {j} ↗"
                if workspace
                else (
                    f"[证据{j}: {_source_label(evidence)} · "
                    f"第{evidence['page']}页 第{evidence['para_id']}段]"
                )
            )
            parts.append(
                f'<a class="trace-citation" href="#{anchor_prefix}ev-{i}-{j}">{escape(label)}</a> '
            )
        if not item["evidences"]:
            parts.append('<p class="trace-warning">未匹配到证据，需人工核查。</p>')
        parts.append("</article>")
    parts.append(
        '</div><div class="trace-sources" aria-label="对应原文证据"><h3>'
        + ("引用证据" if workspace else "原文证据 · 文献与位置")
        + "</h3>"
    )
    for i, item in enumerate(result["sentences"], 1):
        parts.append(
            f'<section id="{anchor_prefix}sentence-{i}" class="trace-evidence-group" tabindex="0">'
            f"<h4>句 {i} 对应的原文片段</h4>"
        )
        if not item["evidences"]:
            parts.append('<p class="trace-warning">未匹配到证据，需人工核查。</p>')
        for j, evidence in enumerate(item["evidences"], 1):
            label = evidence["nli_label"]
            match_detail = (
                f"匹配关系：{MATCH_LABELS[label]}"
                if workspace
                else f"NLI 标签：{label}；模型概率：{evidence['nli_prob']:.3f}"
            )
            status_class, status_text = CORPUS_STATUS[evidence.get("corpus_verified")]
            parts.append(
                f'<article id="{anchor_prefix}ev-{i}-{j}" class="trace-evidence" tabindex="0">'
                '<div class="evidence-badge evidence-source">'
                f"{escape(_source_label(evidence))}</div>"
                '<div class="evidence-badges">'
                f'<span class="evidence-badge">第{evidence["page"]}页</span>'
                f'<span class="evidence-badge">第{evidence["para_id"]}段</span>'
                f'<span class="evidence-badge">匹配置信度 {evidence["nli_prob"]:.1%}</span>'
                f'<span class="evidence-badge {status_class}">{status_text}</span></div>'
                f'<p class="trace-text evidence-excerpt">{escape(evidence["text"])}</p>'
                "<details><summary>查看匹配详情</summary>"
                f'<p class="trace-meta">{match_detail}</p></details></article>'
            )
        parts.append("</section>")
    parts.append("</div></div></section>")
    return "\n".join(parts)


def render_workspace(result: dict, *, anchor_prefix: str) -> str:
    """Render the same evidence data in a reading-focused application shell."""
    return render_provenance(result, anchor_prefix=anchor_prefix, workspace=True)


def _markdown(text: str) -> str:
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", html.escape(text)).replace("\n", "<br>")


def _csv_cell(text: str) -> str:
    # Neutralize spreadsheet formulas, including leading whitespace/control characters.
    if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r")):
        return "'" + text
    return text


def _new_export_folder(output_dir: str | Path | None, prefix: str) -> Path:
    global _EXPORT_DIR
    if output_dir is None:
        with _EXPORT_LOCK:
            if _EXPORT_DIR is None:
                _EXPORT_DIR = tempfile.TemporaryDirectory(prefix="trusted-review-ui-")
            output_dir = _EXPORT_DIR.name
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=output_dir))


def export_review(
    result: dict, output_dir: str | Path | None = None, *, is_demo: bool = False
) -> list[str]:
    """Export the full review independently of evidence reports."""
    from docx import Document
    from docx.oxml.ns import qn
    from docx.shared import Pt

    query = _text(result["query"], "query")
    review = _text(result["review"], "review").replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = review.split("\n\n")
    folder = _new_export_folder(output_dir, "review-")
    md_file = folder / "review.md"
    sections = ["# 文献综述"]
    if is_demo:
        sections.append(f"> {DEMO_NOTICE}")
    sections.extend(["## 研究问题", _markdown(query), "## 综述正文"])
    sections.extend(_markdown(paragraph) for paragraph in paragraphs)
    md_file.write_text("\n\n".join(sections) + "\n", "utf-8")

    document = Document()
    document.core_properties.title = "文献综述"
    for style_name in ("Normal", "Title", "Heading 1"):
        style = document.styles[style_name]
        style.font.name = "Microsoft YaHei"
        style.element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal = document.styles["Normal"]
    normal.font.size = Pt(12)
    normal.paragraph_format.line_spacing = 1.7
    normal.paragraph_format.space_after = Pt(8)
    # Word XML cannot represent some control characters accepted in JSON text.
    invalid_xml = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")
    document.add_heading("文献综述", level=0)
    if is_demo:
        document.add_paragraph(DEMO_NOTICE)
    document.add_heading("研究问题", level=1)
    document.add_paragraph(invalid_xml.sub("\ufffd", query))
    document.add_heading("综述正文", level=1)
    for paragraph in paragraphs:
        document.add_paragraph(invalid_xml.sub("\ufffd", paragraph))
    docx_file = folder / "review.docx"
    document.save(docx_file)
    return [str(md_file), str(docx_file)]


def export_report(
    result: dict,
    output_dir: str | Path | None = None,
    *,
    source_name: str = "导入结果（来源未核验）",
    is_demo: bool = False,
) -> list[str]:
    """Create unique per-request reports; user strings never become paths or HTML."""
    result = validate_result(result)
    source_name = _text(source_name, "source_name")
    if type(is_demo) is not bool:
        raise ValueError("is_demo 必须是布尔值。")
    folder = _new_export_folder(output_dir, "report-")
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
        "| 句号 | 生成句 | 文献名 | 页码 / 段号 | 原文 | NLI 标签 / 模型概率 | 语料一致性校验 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
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
                "语料一致性校验",
            ]
        )
        for i, item in enumerate(result["sentences"], 1):
            evidences = item["evidences"] or [None]
            for evidence in evidences:
                corpus_status = (
                    CORPUS_STATUS[evidence.get("corpus_verified")][1] if evidence else ""
                )
                if evidence is None:
                    cells = [i, item["sentence"], "", "", "未匹配到证据，需人工核查", "", ""]
                    row = (
                        f"| {i} | {_markdown(item['sentence'])} | — | — | "
                        "未匹配到证据，需人工核查 | — | — |"
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
                        f"{evidence['nli_prob']:.3f} | {corpus_status} |"
                    )
                writer.writerow(
                    [
                        _csv_cell(str(cell))
                        for cell in [
                            *cells,
                            source_name,
                            warning,
                            _source_label(evidence) if evidence else "",
                            corpus_status,
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
            raise ReviewBackendError(
                f"生成超过 {timeout} 秒，已停止本次任务；请检查本地模型服务。"
            ) from exc
        except OSError as exc:
            raise ReviewBackendError("无法启动核心 Python；请检查 --backend-python 路径。") from exc
        if completed.returncode:
            reason = completed.stderr.strip()[-3000:] or "核心进程未返回结果。"
            raise ReviewBackendError(f"生成未完成：{reason}")
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
    status = f"{source_name} · {sentence_count} 句综述，{matched_count} 句有对应证据"
    files = export_report(result, source_name=source_name, is_demo=is_demo)
    review_files = export_review(result, is_demo=is_demo)
    rendered = render_workspace(result, anchor_prefix=f"run-{secrets.token_hex(8)}-")
    return status, result["review"], rendered, files, review_files


def _empty_display(status: str) -> tuple:
    return status, "", "", None, None


def _failed(exc: Exception) -> tuple:
    # Clear all old outputs on failure, so an older report cannot look like this run's result.
    return _empty_display(f"未完成：{exc}")


def import_result(path):
    try:
        if not path:
            raise ValueError("请先选择要导入的综述文件。")
        result, metadata = load_result(path)
        # Uploaded result files cannot attest to a check made in this session.
        return _display(check_corpus_evidence(result, None), **metadata)
    except (ValueError, OSError) as exc:
        return _failed(exc)


def show_demo():
    return _display(check_corpus_evidence(DEMO_RESULT, None), "演示样例", is_demo=True)


def generate_review(
    query, pdf_paths, top_k, nli_threshold, *, backend_python=None, corpus_index=None
):
    yield _empty_display("正在生成综述……")
    try:
        result = run_pipeline(query, pdf_paths, top_k, nli_threshold, backend_python=backend_python)
        result = check_corpus_evidence(result, corpus_index)
        names = [Path(path).name for path in _normalize_pdf_paths(pdf_paths)]
        yield _display(result, f"本地生成 · {len(names)} 篇 PDF：" + "、".join(names))
    except ReviewBackendError as exc:
        print(exc, file=sys.stderr)
        yield _empty_display("暂时无法生成综述，请稍后重试。也可以先加载演示样例。")
    except (ValueError, OSError) as exc:
        yield _failed(exc)


def build_app(backend_python=None):
    import gradio as gr

    with gr.Blocks(
        title="可信可溯源科研综述辅助系统",
        fill_width=True,
        css=CSS + UI_CSS,
        theme=gr.themes.Base(primary_hue="blue", secondary_hue="slate", neutral_hue="slate").set(
            block_label_background_fill="transparent",
            block_label_background_fill_dark="transparent",
            block_label_text_color="#5c5e55",
            block_label_text_color_dark="#5c5e55",
            block_label_text_size="12px",
            block_label_text_weight="500",
            input_shadow="none",
            button_primary_background_fill="#245bbb",
            button_primary_text_color="#ffffff",
            button_primary_background_fill_hover="#194993",
        ),
        analytics_enabled=False,
        delete_cache=(3600, 86400),
    ) as demo:
        gr.HTML(
            '<header class="workspace-header"><h1>文献综述工作台</h1></header>',
            elem_id="app-header",
            padding=False,
        )
        with gr.Row(elem_id="workspace-layout"):
            with gr.Column(elem_id="input-panel", min_width=0):
                pdf = gr.File(
                    label="添加文献 · PDF",
                    file_types=[".pdf"],
                    file_count="multiple",
                    height=128,
                    elem_id="pdf-upload",
                )
                query = gr.Textbox(
                    label="研究问题",
                    lines=4,
                    placeholder="你希望从这些文献中了解什么？",
                    elem_id="research-question",
                )
                with gr.Column(elem_id="utility-actions"):
                    submit = gr.Button("生成综述", variant="primary")
                    demo_button = gr.Button("加载演示样例", elem_id="demo-action")
                with gr.Accordion("高级", open=False, elem_id="advanced-panel"):
                    with gr.Accordion("证据溯源报告", open=False, elem_id="report-panel"):
                        files = gr.File(
                            label="Markdown / CSV / JSON / HTML",
                            file_count="multiple",
                            interactive=False,
                        )
                    with gr.Accordion("结果导入", open=False):
                        imported = gr.File(label="综述结果文件（JSON）", file_types=[".json"])
                        import_button = gr.Button("导入并展示")
                    with gr.Accordion("冻结语料", open=False):
                        gr.Markdown("上传后，生成综述时会自动核对引用证据。")
                        corpus_file = gr.File(
                            label="冻结语料文件（JSON）",
                            file_types=[".json"],
                            elem_id="corpus-file",
                        )
                        corpus_status = gr.Textbox(
                            label="冻结语料状态",
                            value="未加载冻结语料。",
                            interactive=False,
                            elem_id="corpus-status",
                        )
                        corpus_state = gr.State()
                    with gr.Accordion("检索与匹配设置", open=False):
                        top_k = gr.Slider(1, 20, value=3, step=1, label="每句候选证据数")
                        threshold = gr.Slider(0, 1, value=0.7, step=0.05, label="最低匹配置信度")
            with gr.Column(elem_id="output-panel", min_width=0):
                with gr.Column(elem_id="reading-toolbar", min_width=0):
                    status = gr.Textbox(
                        label="运行状态与来源",
                        show_label=False,
                        placeholder=" ",
                        interactive=False,
                        lines=1,
                        max_lines=5,
                        elem_id="run-status",
                    )
                    with gr.Row(elem_id="result-actions"):
                        with gr.Accordion("下载综述", open=False, elem_id="export-panel"):
                            review_files = gr.File(
                                label="Markdown / Word", file_count="multiple", interactive=False
                            )
                        with gr.Accordion("综述全文", open=False, elem_id="original-review-panel"):
                            review = gr.Textbox(
                                label="综述全文",
                                interactive=False,
                                lines=5,
                                show_copy_button=True,
                                elem_id="review-full",
                            )
                        clear_button = gr.Button(
                            "清空结果", elem_id="clear-action", scale=0, min_width=84
                        )
                gr.HTML(value=EMPTY_WORKSPACE, padding=False, elem_id="empty-reader")
                provenance = gr.HTML(label="逐句证据溯源", padding=False, elem_id="provenance")
        outputs = [status, review, provenance, files, review_files]
        shared = {"outputs": outputs, "concurrency_id": "review-actions", "concurrency_limit": 1}

        def generate(query, pdf, top_k, threshold, corpus_index):
            yield from generate_review(
                query,
                pdf,
                top_k,
                threshold,
                backend_python=backend_python,
                corpus_index=corpus_index,
            )

        submit.click(
            generate,
            [query, pdf, top_k, threshold, corpus_state],
            api_name="generate_review",
            **shared,
        )
        import_button.click(import_result, [imported], api_name="import_result", **shared)
        demo_button.click(show_demo, api_name="show_demo", **shared)
        clear_button.click(
            lambda: _empty_display("已清空结果。"), api_name="clear_result", **shared
        )

        corpus_file.change(
            load_frozen_corpus,
            [corpus_file],
            [corpus_state, corpus_status],
            api_name="load_frozen_corpus",
            concurrency_id="review-actions",
            concurrency_limit=1,
        )

        def diagnose() -> str:
            return check_environment(backend_python)

        gr.api(
            diagnose,
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
