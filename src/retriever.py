import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


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

    def search(self, query, top_k=3):
        """返回 top_k 个段落（带元数据）和对应相似度"""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "text": self.corpus[idx]["text"],
                "page": self.corpus[idx]["page"],
                "para_id": self.corpus[idx]["para_id"],
                "score": float(similarities[idx])
            })
        return results