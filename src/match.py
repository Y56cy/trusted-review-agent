"""句级证据匹配模块（后续实现）。

职责（对应申请书 模块3 / 技术路线-验证层）：
  1. 将生成的综述文本拆分为独立语句
  2. 向量相似度粗筛 + 自然语言推理（NLI）语义关系精筛，匹配到原始文献片段
  3. 记录文献名 / 页码 / 片段编号，支撑溯源展示与评价指标计算

TODO（待决策返回后再实现）：
  - [核心风险] NLI 做"句↔证据片段"匹配的可行性未验证，需先做 spike（评审问题 4）。
    这是全项目唯一创新点，最应先验证。
  - [评价指标] 证据覆盖率 / 引文一致性 / 文本忠实度的计算公式未定（评审问题 1）。
  - [性能] NLI 精筛是 O(句数 × 候选数 × 推理) 的性能悬崖，需实测延迟（评审问题 13）。
"""
from __future__ import annotations

import re

import torch
from transformers import BertForSequenceClassification, BertTokenizer

# 一句 = 若干非句末标点字符 + 至少一个句末标点；末尾无标点的残余也算一句。
_SENT_END_RE = re.compile(r"[^。！？!?]*[。！？!?]+|[^。！？!?]+$")

# NLI 标签顺序（已实测确认）。加载必须用 Bert*，不要用 AutoTokenizer（README 决策）。
_NLI_LABELS = ["contradiction", "neutral", "entailment"]
_NLI_PATH = r"D:\dachuang\trusted-review-agent\models\Fengshenbang--Erlangshen-RoBERTa-110M-NLI\snapshots\master"

# NLI 模型较大，懒加载：首次调用才加载，避免 import match 就触发重模型。
_nli_tokenizer = None
_nli_model = None


def _load_nli():
    global _nli_tokenizer, _nli_model
    if _nli_model is None:
        _nli_tokenizer = BertTokenizer.from_pretrained(_NLI_PATH)
        _nli_model = BertForSequenceClassification.from_pretrained(_NLI_PATH)
        _nli_model.eval()
    return _nli_tokenizer, _nli_model


def nli_predict(premise: str, hypothesis: str) -> tuple[str, float]:
    """对「证据(premise) → 生成句(hypothesis)」做蕴含判断，返回 (label, prob)。"""
    tokenizer, model = _load_nli()
    inputs = tokenizer(premise, hypothesis, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    pred_idx = int(torch.argmax(probs).item())
    return _NLI_LABELS[pred_idx], float(probs[pred_idx].item())


def split_sentences(review_text: str) -> list[str]:
    """将综述文本拆分为独立语句（保留句末标点）。

    按中文句末标点（。！？）及英文问号/叹号切分，连续标点（如"！！"）算作同一句。
    英文句点 "." 暂不参与切分，避免把小数（3.14）或缩写（e.g.）切开；综述生成句
    以中文标点为主，够用。
    """
    parts = _SENT_END_RE.findall(review_text)
    return [p.strip() for p in parts if p.strip()]


def match_sentence(sentence: str, candidates: list[dict], threshold: float = 0.7) -> list[dict]:
    """对单句做 NLI 精筛，返回命中的证据片段。

    candidates 是 retriever.search() 返回的候选段落（向量粗筛已在那里完成）。
    对每个候选做 NLI（premise=证据, hypothesis=生成句），仅保留 entailment 且
    概率 ≥ threshold 的段落作为命中证据（决策 2）。
    """
    matches = []
    for c in candidates:
        label, prob = nli_predict(c["text"], sentence)
        if label == "entailment" and prob >= threshold:
            matches.append({
                "para_id": c["para_id"],
                "page": c["page"],
                "text": c["text"],
                "nli_label": label,
                "nli_prob": prob,
            })
    return matches


def compute_metrics(matches: list[dict], review_text: str) -> dict:
    """计算证据覆盖率 / 引文一致性 / 文本忠实度。公式待定。"""
    raise NotImplementedError("待评价指标定义后实现")
