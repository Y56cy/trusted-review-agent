"""检索模块：学术 PDF 解析、细粒度分块与混合检索。

职责（对应申请书 模块1 / 模块2 / 技术路线-数据层-检索层）：
  1. 解析学术 PDF，按"篇章—段落—论据句"层级分块，保留文献名、页码、片段编号等元数据
  2. 构建 BGE 稠密向量索引 + BM25 关键词索引
  3. 双路召回 + 相关性筛选，为生成模块输出证据片段

TODO（待 5 项高优先级决策返回后再实现，未定项不在此拍板）：
  - [数据模型] 检索单元 = 论据句 还是 段落，未定（评审问题 8）。
  - [PDF 解析] 解析工具未定。当前依赖里已有 pypdf（简单），但复杂版式/扫描件可能需 pymupdf 或 MinerU（评审问题 3/5）。
  - [向量库] 向量存储选型未定（FAISS / Chroma / numpy 自算），当前依赖里尚无 faiss / chromadb。
"""
from __future__ import annotations

from pathlib import Path


def build_index(papers_dir: str | Path) -> None:
    """解析目录下的 PDF，分块、抽取元数据，建立向量 + BM25 索引。

    索引结果落盘到 models/ 或 results/（待定）。
    """
    raise NotImplementedError("待数据模型 / PDF 解析工具 / 向量库选型决策后实现")


def retrieve(query: str, top_k: int = 10) -> list[dict]:
    """双路召回：BGE 向量检索 + BM25 关键词检索，融合后返回证据片段。

    返回格式（建议，可改）：
        [{"chunk_id": str, "text": str, "doc": str, "page": int, "score": float}, ...]
    """
    raise NotImplementedError("待检索层选型决策后实现")
