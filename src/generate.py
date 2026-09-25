"""生成模块：本地量化大模型的综述初稿生成。

职责（对应申请书 模块2 / 技术路线-生成层）：
  1. 接收用户查询主题 + 检索层返回的证据片段
  2. 通过提示词编排，引导模型按"研究背景—研究现状—方法梳理—主要问题—未来展望"结构组织文本
  3. 本地推理（减少对外部云端 API 的依赖）

TODO（待决策返回后再实现）：
  - [硬件/模型] 目标硬件与量化档位未定。注意：当前依赖里 torch 为 CPU 版（torch==2.14.0+cpu），
    尚无 llama-cpp-python / gguf 等本地推理后端，7B 模型如何跑需先定（评审问题 7）。
  - [引文] 生成层是否输出引用标记（如 [1][2]）未定，直接影响"引文一致性"指标能否计算（评审问题 10）。
  - [生成质量] 7B 量化模型的综述质量未验证，建议先做一次真实文献的生成实验（评审问题 16）。
"""
from __future__ import annotations


REVIEW_STRUCTURE = ["研究背景", "研究现状", "方法梳理", "主要问题", "未来展望"]


def generate_review(query: str, evidence_chunks: list[dict]) -> str:
    """基于查询与证据片段，生成本地综述初稿文本。"""
    raise NotImplementedError("待硬件 / 模型 / 引文机制决策后实现")


def build_prompt(query: str, evidence_chunks: list[dict]) -> str:
    """构造结构化提示词（结构参照 REVIEW_STRUCTURE，证据片段作为主要上下文）。"""
    raise NotImplementedError("待生成层设计决策后实现")
