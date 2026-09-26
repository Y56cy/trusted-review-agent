from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 1. 加载 BGE 中文向量模型（第一次运行会自动下载，约 400MB）
model = SentenceTransformer("../models/bge-small-zh-v1.5")

# 2. 准备几段测试文本（模拟从 PDF 里解析出来的文献片段）
corpus = [
    "大语言模型在科研辅助中的应用越来越广泛，尤其是在文献检索和综述生成方面。",
    "检索增强生成（RAG）通过引入外部知识库，可以有效减少大模型的幻觉问题。",
    "句级证据匹配是指将生成文本拆分为句子，并与原始文献片段建立对应关系。",
    "BM25 是一种经典的关键词检索算法，常用于信息检索系统中的文本召回。",
    "本地化部署可以减少对外部 API 的依赖，适合高校图书馆的内网环境。",
]

# 3. 把所有片段编码成向量
corpus_embeddings = model.encode(corpus, normalize_embeddings=True)

# 4. 输入一个查询
query = "怎么减少大模型编造内容？"
query_embedding = model.encode([query], normalize_embeddings=True)

# 5. 计算相似度并排序
similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
ranked = np.argsort(similarities)[::-1]

print(f"查询：{query}\n")
for idx in ranked:
    print(f"相似度 {similarities[idx]:.4f} | {corpus[idx]}")