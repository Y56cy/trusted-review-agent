import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def _chinese_ratio(text):
    """中文字符在非空白字符中的占比，用于过滤英文段落。

    BGE 是中文向量模型，对英文段落编码能力有限，但英文段落仍可能以
    0.5+ 的相似度混入 Top-K。中文字符占比低于阈值的段落视为英文段落剔除。
    """
    non_space = re.sub(r'\s', '', text)
    if not non_space:
        return 0.0
    return len(re.findall(r'[一-鿿]', non_space)) / len(non_space)


class Retriever:
    def __init__(self, model_path="../models/bge-small-zh-v1.5"):
        print("加载 BGE 模型...")
        self.model = SentenceTransformer(model_path)
        self.corpus = []
        self.embeddings = None

    def build_index(self, paragraphs):
        """paragraphs 是 parser.parse_pdf() 返回的段落列表"""
        self.corpus = paragraphs
        texts = [p["text"] for p in paragraphs]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query, top_k=3, min_chinese_ratio=0.3):
        """返回 top_k 个段落（带元数据）和对应相似度。

        先按相似度多取候选，再剔除中文字符占比低于 min_chinese_ratio 的英文
        段落（参考文献、英文图表标题等），最后返回剩余中的 top_k。英文段落
        过多时可能返回不足 top_k 条，属正常现象。
        """
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        # 多取 3 倍候选，给英文过滤留出余量
        n_candidates = min(len(self.corpus), top_k * 3)
        top_indices = np.argsort(similarities)[::-1][:n_candidates]
        results = []
        for idx in top_indices:
            if _chinese_ratio(self.corpus[idx]["text"]) < min_chinese_ratio:
                continue
            results.append({
                "text": self.corpus[idx]["text"],
                "page": self.corpus[idx]["page"],
                "para_id": self.corpus[idx]["para_id"],
                "score": float(similarities[idx])
            })
            if len(results) >= top_k:
                break
        return results