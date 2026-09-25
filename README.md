# 基于智能体的可信可溯源科研综述辅助系统
面向高校图书馆科研服务场景的本地化科研综述辅助原型。核心链路：

`文献解析 → 证据检索 → 综述生成 → 句级校验 → 溯源展示 → 效果评价`

## 目录结构

```
├── venv/                # 虚拟环境（自动生成，勿提交）
├── requirements.txt     # 依赖清单
├── src/
│   ├── retrieve.py      # 检索模块：PDF 解析 + 分块 + 混合检索
│   ├── generate.py      # 生成模块：本地量化大模型的综述初稿生成
│   └── match.py         # 句级证据匹配模块（后续实现）
├── data/papers/         # 测试用 PDF 文献
├── models/              # 下载的 BGE 等模型权重
├── results/             # 实验记录、输出结果
└── README.md
```


## 环境要求
- Python 3.10
- llama.cpp（CPU 版即可跑通，GPU 版可选）
- Qwen2.5-7B-Instruct Q4_K_M GGUF 模型

## 安装
pip install -r requirements.txt

## 模型下载
- BGE 向量模型：通过 ModelScope 下载 `BAAI/bge-small-zh-v1.5` 到 `models/bge-small-zh-v1.5`
- LLM：下载 Qwen2.5-7B-Instruct Q4_K_M GGUF 到 `D:\llama`（或任意路径）

## 运行
1. 启动 llama-server：
   .\llama-server.exe -m qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -ngl 0 --port 8080
2. 运行检索脚本：
   python src/test_retrieve.py
3. 运行 RAG 脚本：
   python src/test_rag.py

## 当前进度
- [x] 本地大模型推理跑通
- [x] BGE 向量检索跑通
- [x] 检索 + 生成 RAG 链路跑通
- [ ] PDF 解析
- [ ] 句级证据匹配
- [ ] 溯源展示
- [ ] Gradio 界面

