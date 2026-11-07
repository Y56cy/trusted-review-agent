"""BGE / BM25 检索；默认使用等权 RRF，score 始终保留余弦相似度语义。"""

import math
import re
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models/bge-small-zh-v1.5"
METHODS = ("vector", "bm25", "hybrid")
RRF_K = 60
TOKENIZER = jieba.Tokenizer()


def _chinese_ratio(text):
    non_space = re.sub(r"\s", "", text)
    if not non_space:
        return 0.0
    return len(re.findall(r"[\u4e00-\u9fff]", non_space)) / len(non_space)


def tokenize(text):
    """固定 jieba 词典、HMM=False；英文小写，保留中文、字母和数字片段。"""
    return [
        token
        for word in TOKENIZER.cut(text.lower(), HMM=False)
        for token in re.findall(r"[\u4e00-\u9fff]+|[a-z0-9]+", word)
    ]


def _load_embedding_model(model_path):
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"BGE 本地模型不存在：{model_path}。请通过 ModelScope 下载 "
            "BAAI/bge-small-zh-v1.5，或显式传入 model_path。"
        )
    # 延迟导入，让轻量逻辑测试不必安装或加载 torch / transformers。
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(model_path), local_files_only=True)


def _normalize(vectors):
    vectors = np.asarray(vectors, dtype=float)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("向量必须是有限数值组成的二维矩阵")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


class Retriever:
    def __init__(self, model_path=None):
        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else DEFAULT_MODEL_PATH
        )
        self.model = _load_embedding_model(self.model_path)
        self.corpus = []
        self.embeddings = None
        self.bm25 = None
        self.tokens = []
        self.chinese_ratios = []

    def build_index(self, paragraphs):
        """原子替换双索引。输入采用 parser 的 text / page / para_id 格式。"""
        corpus = []
        ids = set()
        for p in paragraphs:
            if not isinstance(p, dict) or not isinstance(p.get("text"), str):
                raise ValueError("段落必须包含 text 字符串")
            if not p["text"].strip():
                raise ValueError("段落 text 不能为空")
            for key in ("page", "para_id"):
                if type(p.get(key)) is not int or p[key] <= 0:
                    raise ValueError(f"{key} 必须是正整数")
            if p["para_id"] in ids:
                raise ValueError(f"para_id 重复：{p['para_id']}")
            ids.add(p["para_id"])
            corpus.append({k: p[k] for k in ("text", "page", "para_id")})
        if not corpus:
            raise ValueError("无有效语料，无法建立索引")
        tokens = [tokenize(p["text"]) for p in corpus]
        if not any(tokens):
            raise ValueError("语料不包含可索引的中文、字母或数字")
        bm25 = BM25Okapi(tokens, k1=1.5, b=0.75)
        embeddings = _normalize(
            self.model.encode([p["text"] for p in corpus], normalize_embeddings=True)
        )
        if len(embeddings) != len(corpus):
            raise ValueError("模型返回的向量数与段落数不一致")
        self.corpus, self.tokens = corpus, tokens
        self.bm25, self.embeddings = bm25, embeddings
        self.chinese_ratios = [_chinese_ratio(p["text"]) for p in corpus]

    def search(self, query, top_k=3, min_chinese_ratio=0.3, *, method="hybrid"):
        """返回原有四字段。混合顺序由 RRF 决定，不可按 score 再排序。

        score 在所有模式中均为 BGE 余弦相似度。BM25 仅返回正分命中；
        hybrid 无关键词命中时退回 vector。过滤先于各通道截取候选。
        """
        if type(top_k) is not int or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        if method not in METHODS:
            raise ValueError(f"method 必须为 {METHODS}")
        if (
            isinstance(min_chinese_ratio, bool)
            or not isinstance(min_chinese_ratio, (int, float))
            or not math.isfinite(min_chinese_ratio)
            or not 0 <= min_chinese_ratio <= 1
        ):
            raise ValueError("min_chinese_ratio 必须在 0 到 1 之间")
        if not isinstance(query, str):
            raise ValueError("query 必须是字符串")
        if self.embeddings is None:
            raise RuntimeError("请先调用 build_index 建立索引")
        if not query.strip():
            return []
        eligible = [i for i, ratio in enumerate(self.chinese_ratios) if ratio >= min_chinese_ratio]
        if not eligible:
            return []
        embedding = _normalize(self.model.encode([query], normalize_embeddings=True))
        similarities = (embedding @ self.embeddings.T)[0]
        vector = sorted(eligible, key=lambda i: (-similarities[i], self.corpus[i]["para_id"]))
        if method == "vector":
            ranked = vector
        else:
            scores = self.bm25.get_scores(tokenize(query))
            lexical = sorted(
                (i for i in eligible if scores[i] > 0),
                key=lambda i: (-scores[i], -similarities[i], self.corpus[i]["para_id"]),
            )
            if method == "bm25":
                ranked = lexical
            elif not lexical:
                ranked = vector
            else:
                limit = min(len(eligible), max(20, top_k))
                fused = {}
                for channel in (vector[:limit], lexical[:limit]):
                    for rank, i in enumerate(channel, 1):
                        fused[i] = fused.get(i, 0.0) + 0.5 / (RRF_K + rank)
                ranked = sorted(
                    fused, key=lambda i: (-fused[i], -similarities[i], self.corpus[i]["para_id"])
                )
        return [dict(self.corpus[i], score=float(similarities[i])) for i in ranked[:top_k]]
