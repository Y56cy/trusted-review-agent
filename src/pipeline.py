"""端到端 pipeline：文献解析 → 检索 → 生成 → 句级校验 → 溯源展示。

把 parser / retriever / generator / match 四个模块串起来，是中期检查原型的
唯一入口。Day 3 的 Gradio 界面直接 import 这里的 run_review()。
"""
from __future__ import annotations

from pathlib import Path

from parser import parse_pdf
from retriever import Retriever
from generator import generate
from match import split_sentences, match_sentence


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


def run_review(query, pdf_paths, top_k=3, nli_threshold=0.7):
    """跑通全链路，返回结构化结果。

    pdf_paths 支持单个路径字符串或路径列表；列表即「多篇文献库」场景。

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

    # 4. 拆句 + 逐句证据匹配
    sentences = split_sentences(review)
    sentence_results = []
    for sent in sentences:
        candidates = retriever.search(sent, top_k=top_k)
        evidences = match_sentence(sent, candidates, threshold=nli_threshold)
        sentence_results.append({"sentence": sent, "evidences": evidences})

    return {"query": query, "review": review, "sentences": sentence_results}


def _print_result(result):
    """命令行友好打印（中期原型用）"""
    print("=" * 60)
    print("用户问题：", result["query"])
    print("=" * 60)
    print("\n【生成的综述】")
    print(result["review"])
    print("\n【逐句证据匹配】")
    for i, item in enumerate(result["sentences"], 1):
        print(f"\n[句{i}] {item['sentence']}")
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

    query = "检索增强生成（RAG）的核心框架、关键技术与发展现状是什么？"
    result = run_review(query, pdf_paths)
    _print_result(result)
