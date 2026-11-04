import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from parser import parse_pdf
from retriever import Retriever
from generator import generate


# 1. 解析 PDF
pdf_path = "../data/papers/基于双边网格滤波的3D高斯辐射场无人机SLAM方法_高连科.pdf"
paragraphs = parse_pdf(pdf_path)
print(f"有效段落数：{len(paragraphs)}")

# 2. 建立检索索引
retriever = Retriever()
retriever.build_index(paragraphs)

# 3. 用户查询
query = "这篇论文提出了什么方法来解决光照变化问题？"
results = retriever.search(query, top_k=3)

print("\n检索到的证据：")
for i, item in enumerate(results):
    print(f"  [{i+1}] 第{item['page']}页 第{item['para_id']}段 | 相似度 {item['score']:.4f}")
    print(f"      {item['text'][:120]}...")

# 4. 拼接 prompt 并生成
context = "\n".join([
    f"[{i+1}] （第{item['page']}页 第{item['para_id']}段）{item['text']}"
    for i, item in enumerate(results)
])

prompt = f"""你是一个科研综述辅助助手。请根据以下证据片段，回答用户的问题。

证据：
{context}

用户问题：{query}

请基于上述证据，用一段话回答，并在句末标注证据编号（如 [1]）。"""

print("\n正在调用本地大模型...")
answer = generate(prompt)
print("\n=== 大模型回答 ===")
print(answer)