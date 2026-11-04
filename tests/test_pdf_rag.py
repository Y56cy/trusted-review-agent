import re
import numpy as np
import requests
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ========== 1. 清洗函数（保持不变） ==========
def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    lines = text.split('\n')
    merged = []
    buffer = ""
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) <= 2:
            buffer += line
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(line)
    if buffer:
        merged.append(buffer)
    return '\n'.join(merged)


# ========== 2. 按页解析 PDF，保留页码和段落编号 ==========
reader = PdfReader("../data/papers/基于双边网格滤波的3D高斯辐射场无人机SLAM方法_高连科.pdf")

all_paragraphs = []   # 每个元素是 {"text": ..., "page": ..., "para_id": ...}
para_counter = 0

for page_num, page in enumerate(reader.pages, start=1):
    page_text = page.extract_text() + "\n"
    cleaned = clean_text(page_text)
    paras = [p.strip() for p in cleaned.split('\n') if len(p.strip()) >= 30]
    for p in paras:
        para_counter += 1
        all_paragraphs.append({
            "text": p,
            "page": page_num,
            "para_id": para_counter
        })

print(f"有效段落数：{len(all_paragraphs)}")


# ========== 3. 建立 BGE 索引 ==========
print("加载 BGE 模型...")
retriever = SentenceTransformer("../models/bge-small-zh-v1.5")

texts = [p["text"] for p in all_paragraphs]
corpus_embeddings = retriever.encode(texts, normalize_embeddings=True)


# ========== 4. 用户查询 ==========
query = "这篇论文提出了什么方法来解决光照变化问题？"
query_embedding = retriever.encode([query], normalize_embeddings=True)


# ========== 5. 检索 Top-3，返回带元数据的段落 ==========
similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
top_k = np.argsort(similarities)[::-1][:3]
retrieved = [all_paragraphs[i] for i in top_k]

print("\n检索到的证据：")
for i, item in enumerate(retrieved):
    print(f"  [{i+1}] 第{item['page']}页 第{item['para_id']}段 | 相似度 {similarities[top_k[i]]:.4f}")
    print(f"      {item['text'][:120]}...")


# ========== 6. 调用本地大模型 ==========
context = "\n".join([
    f"[{i+1}] （第{item['page']}页 第{item['para_id']}段）{item['text']}"
    for i, item in enumerate(retrieved)
])

prompt = f"""你是一个科研综述辅助助手。请根据以下证据片段，回答用户的问题。

证据：
{context}

用户问题：{query}

请基于上述证据，用一段话回答，并在句末标注证据编号（如 [1]）。"""

print("\n正在调用本地大模型...")
response = requests.post(
    "http://127.0.0.1:8080/v1/chat/completions",
    json={
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 512
    }
)

result = response.json()
print("\n=== 大模型回答 ===")
print(result["choices"][0]["message"]["content"])