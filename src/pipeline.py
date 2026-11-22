"""端到端 pipeline：文献解析 → 检索 → 生成 → 句级校验 → 溯源展示。

把 parser / retriever / generator / match 四个模块串起来，是中期检查原型的
唯一入口。Day 3 的 Gradio 界面直接 import 这里的 run_review()。
"""
from __future__ import annotations

import re
from pathlib import Path

from generator import UNSUPPORTED, generate, rewrite_sentence
from match import match_sentence, split_sentences
from parser import parse_pdf
from retriever import Retriever


def _normalize_paths(pdf_paths):
    """把单个路径或路径列表统一成列表。"""
    if isinstance(pdf_paths, (str, Path)):
        return [pdf_paths]
    return list(pdf_paths)


def _build_prompt(query: str, context_paras: list[dict]) -> str:
    context = "\n".join(
        f"[{i+1}] （{item['source']} 第{item['page']}页 第{item['para_id']}段）{item['text']}"
        for i, item in enumerate(context_paras)
    )
    return f"""你是一个科研综述辅助助手。请根据以下证据片段，用 3-6 句话的中文综述回答用户的问题，要求客观、通顺、只依据证据内容，不要编造。

证据：
{context}

用户问题：{query}
"""


def run_review(query, pdf_paths, top_k=3, nli_threshold=0.7, trace=None):
    """跑通全链路，返回结构化结果。

    pdf_paths 支持单个路径字符串或路径列表；列表即「多篇文献库」场景。
    trace 为可选的 list，传入后记录无证据句的改写轨迹（仅 CLI 调试用，不影响返回结果）。

    返回 dict：
      {
        "query": str,
        "review": str,                # 生成的综述全文
        "sentences": [                # 每句 + 命中的证据
            {"sentence": str, "evidences": [{"para_id","page","source","text","nli_label","nli_prob"}]},
            ...
        ]
      }
    """
    # 1. 解析（多篇）PDF，合并为语料
    paragraphs = []
    for path in _normalize_paths(pdf_paths):
        paragraphs.extend(parse_pdf(path))

    # 2. 建索引
    retriever = Retriever()
    retriever.build_index(paragraphs)

    # 3. 检索 + 生成综述（RAG）
    context_paras = retriever.search(query, top_k=top_k)
    review = generate(_build_prompt(query, context_paras))

    # 4. 拆句 + 逐句证据匹配 + 无证据句改写闭环
    sentences = split_sentences(review)
    sentence_results = []
    rewrote = False
    for sent in sentences:
        candidates = retriever.search(sent, top_k=top_k)
        evidences = match_sentence(sent, candidates, threshold=nli_threshold)
        if evidences:
            sentence_results.append({"sentence": sent, "evidences": evidences})
            continue

        # 无证据句：交给模型改写，改写后再跑一遍 NLI 校验。
        # 改不动（仍无证据，或模型明确返回 UNSUPPORTED）就保持原句、空证据，即「未能溯源」。
        try:
            rewritten = rewrite_sentence(sent, candidates)
        except Exception:
            _append_trace(trace, "error", sent)
            sentence_results.append({"sentence": sent, "evidences": []})
            continue
        if rewritten == UNSUPPORTED:
            _append_trace(trace, "unsupported", sent)
            sentence_results.append({"sentence": sent, "evidences": []})
            continue
        new_candidates = retriever.search(rewritten, top_k=top_k)
        new_evidences = match_sentence(rewritten, new_candidates, threshold=nli_threshold)
        if new_evidences:
            _append_trace(trace, "rewritten", sent, rewritten)
            sentence_results.append({"sentence": rewritten, "evidences": new_evidences})
            rewrote = True
        else:
            _append_trace(trace, "still_unsupported", sent, rewritten)
            sentence_results.append({"sentence": sent, "evidences": []})

    # 5. 有改写时，用最终句子重建综述全文，保证改写句同步回 review
    if rewrote:
        review = "".join(item["sentence"] for item in sentence_results)

    return {"query": query, "review": review, "sentences": sentence_results}


_PLACEHOLDER_RE = re.compile(r"<K(\d+)>")


def expand_review(result):
    """把可溯源关键句扩写成结构化综述（引言/过渡/主体/总结）。

    占位符法：LLM 只写结构句 + <K1>..<Kn> 占位符，关键句由代码逐字插回，
    保证关键句的溯源锚点绝不被改写。结构句标 structural=True，关键句标 False。
    """
    key_sentences = [item for item in result["sentences"] if item["evidences"]]
    if not key_sentences:
        return result

    numbered = "\n".join(
        f"[关键句{i}] {item['sentence']}" for i, item in enumerate(key_sentences, 1)
    )
    placeholders = "、".join(f"<K{i}>" for i in range(1, len(key_sentences) + 1))
    prompt = f"""你是综述扩写智能体。以下是已生成的可溯源关键句（每句都有原文证据支撑）：

{numbered}

请把这些关键句扩写成一篇完整的学术综述，结构为：引言 → 关键句 → 过渡 → 关键句 → 总结。规则：
1. 不要改动任何关键句的文字，也不要添加新的证据性句子；
2. 用占位符 {placeholders} 表示「这里放对应的关键句」；
3. 引言、过渡、总结都要写完整的实际句子（引言点明主题、过渡衔接上下文、总结概括要点），禁止输出「引言」「过渡」「总结」这类占位词；
4. 只输出引言/过渡/总结的句子和占位符，不要输出任何解释。

示例结构（「……」表示你要写的实际内容，<K1> 是占位符）：
……。<K1>……。<K2>……。"""

    scaffold = generate(prompt).strip()

    sentences = []
    used = set()
    last = 0
    for m in _PLACEHOLDER_RE.finditer(scaffold):
        for s in split_sentences(scaffold[last : m.start()]):
            sentences.append({"sentence": s, "evidences": [], "structural": True})
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(key_sentences):
            sentences.append(
                {
                    "sentence": key_sentences[idx]["sentence"],
                    "evidences": key_sentences[idx]["evidences"],
                    "structural": False,
                }
            )
            used.add(idx)
        last = m.end()
    for s in split_sentences(scaffold[last:]):
        sentences.append({"sentence": s, "evidences": [], "structural": True})

    # 兜底：补上 LLM 漏掉占位符而未引用的关键句
    for i, item in enumerate(key_sentences):
        if i not in used:
            sentences.append(
                {
                    "sentence": item["sentence"],
                    "evidences": item["evidences"],
                    "structural": False,
                }
            )

    review = "".join(s["sentence"] for s in sentences)
    return {"query": result["query"], "review": review, "sentences": sentences}


def _append_trace(trace, kind, original, rewritten=None):
    """可选地记录一条改写轨迹；trace 为 None 时不做任何事（前端调用不传 trace）。"""
    if trace is None:
        return
    entry = {"kind": kind, "original": original}
    if rewritten is not None:
        entry["rewritten"] = rewritten
    trace.append(entry)


def _print_trace_entry(entry):
    kind = entry["kind"]
    original = entry["original"]
    if kind == "rewritten":
        print(f"  改写：{original}\n       → {entry['rewritten']}")
    elif kind == "unsupported":
        print(f"  未能溯源（候选证据不足以支撑）：{original}")
    elif kind == "still_unsupported":
        print(f"  改写后仍无证据，保持原句：{original}\n       （改写尝试：{entry['rewritten']}）")
    elif kind == "error":
        print(f"  改写异常，保持原句：{original}")


def _print_result(result, trace=None):
    """命令行友好打印（中期原型用）"""
    print("=" * 60)
    print("用户问题：", result["query"])
    print("=" * 60)
    print("\n【生成的综述】")
    print(result["review"])
    if trace:
        print("\n【改写轨迹】")
        for entry in trace:
            _print_trace_entry(entry)
    print("\n【逐句证据匹配】")
    for i, item in enumerate(result["sentences"], 1):
        print(f"\n[句{i}] {item['sentence']}")
        if item.get("structural"):
            print("   （综述性扩写，非直接引用）")
            continue
        if not item["evidences"]:
            print("   ⚠️ 无命中证据")
            continue
        for ev in item["evidences"]:
            print(f"   → 证据: 《{ev.get('source', '')}》 第{ev['page']}页 第{ev['para_id']}段 "
                  f"({ev['nli_label']}, {ev['nli_prob']:.2f})")
            print(f"     原文: {ev['text'][:80]}...")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    repo_root = Path(__file__).resolve().parent.parent
    pdf_paths = [str(p) for p in sorted((repo_root / "data" / "corpus").glob("*.pdf"))]
    if not pdf_paths:
        sys.exit("未找到语料：请把 PDF 放入 data/corpus/ 后重试")

    query = "向量检索和 BM25 关键词检索在 RAG 里是怎么配合工作的？" #检索增强生成（RAG）的核心框架、关键技术与发展现状是什么？
    trace = []
    result = run_review(query, pdf_paths, trace=trace)
    _print_result(result, trace)

    print("\n" + "=" * 60)
    print("【综述扩写智能体：扩写后的完整综述】")
    print("=" * 60)
    expanded = expand_review(result)
    _print_result(expanded)

#     按触发概率从高到低：

#   1. RAG 相比纯大模型直接生成回答，核心优势是什么？（"优势"容易让 LLM 跨段综合）
#   2. 检索增强生成是如何缓解大模型幻觉的？（"如何缓解"容易绕开原文措辞）
#   3. 向量检索和 BM25 关键词检索在 RAG 里是怎么配合工作的？（混合检索是语料里有、但原文用
#   RRF 等术语，LLM 可能换说法）
#   RAG系统在医疗影像领域的落地案例有什么？