import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import json

# 1. 加载 BGE 检索模型
print("加载 BGE 模型...")
retriever = SentenceTransformer("../models/bge-small-zh-v1.5")

# 2. 模拟文献库
corpus = [
    "大语言模型在科研辅助中的应用越来越广泛，尤其是在文献检索和综述生成方面。",
    "检索增强生成（RAG）通过引入外部知识库，可以有效减少大模型的幻觉问题。",
    "句级证据匹配是指将生成文本拆分为句子，并与原始文献片段建立对应关系。",
    "BM25 是一种经典的关键词检索算法，常用于信息检索系统中的文本召回。",
    "本地化部署可以减少对外部 API 的依赖，适合高校图书馆的内网环境。",
]

# 3. 建立向量索引
corpus_embeddings = retriever.encode(corpus, normalize_embeddings=True)

# 4. 用户查询
query = "怎么减少大模型编造内容？"
query_embedding = retriever.encode([query], normalize_embeddings=True)

# 5. 检索 Top-3 相关片段
similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
top_k = np.argsort(similarities)[::-1][:3]
retrieved = [corpus[i] for i in top_k]

print("\n检索到的证据：")
for i, r in enumerate(retrieved):
    print(f"  [{i+1}] {r}")

# 6. 拼接 prompt，发给本地大模型
context = "\n".join([f"[{i+1}] {r}" for i, r in enumerate(retrieved)])
prompt = f"""你是一个科研综述辅助助手。请根据以下证据片段，回答用户的问题。

证据：
{context}

用户问题：{query}

请基于上述证据，用一段话回答，并在句末标注证据编号（如 [1]）。如果证据不足以回答，请说明。"""

print("\n正在调用本地大模型生成回答...")

response = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={
        "model": "qwen",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 512
    }
)

result = response.json()
answer = result["choices"][0]["message"]["content"]

print("\n=== 大模型回答 ===")
print(answer)