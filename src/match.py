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


def split_sentences(review_text: str) -> list[str]:
    """将综述文本拆分为独立语句。"""
    raise NotImplementedError("待实现")


def match_sentence(sentence: str, candidates: list[dict], top_k: int = 3) -> list[dict]:
    """对单句做"向量粗筛 + NLI 精筛"，返回命中的证据片段。"""
    raise NotImplementedError("待 NLI 可行性 spike 后再实现")


def compute_metrics(matches: list[dict], review_text: str) -> dict:
    """计算证据覆盖率 / 引文一致性 / 文本忠实度。公式待定。"""
    raise NotImplementedError("待评价指标定义后实现")
