# 基于智能体的可信可溯源科研综述辅助系统
面向高校图书馆科研服务场景的本地化科研综述辅助原型。核心链路：

`文献解析 → 证据检索 → 综述生成 → 句级校验 → 溯源展示 → 效果评价`

## 项目定位（重要）

本项目是**工程流程整合**，不追求底层算法创新。不重新训练大模型，不提出新生成算法，而是把 RAG、本地大模型、文本匹配技术整合成高校图书馆场景下的可追溯工作流。

**句级证据匹配只用于提供"证据支持与人工核查依据"，不声称能自动证明结论为真。** 这是已确定的边界，后续实现不得越过。

## 新成员上手（先看这里）

1. 看 `docs/分工.md` —— 知道自己负责哪个模块、该改哪个文件
2. 看 `docs/执行方案.md` —— 了解分几个阶段、每个阶段交付什么
3. 看 `docs/中期冲刺计划.md` —— 知道中期检查前要先做什么

代码入口是 `src/pipeline.py`，一条命令跑通全链路。团队进度以这三份文档的 checkbox 为准。

## 目录结构

```
├── venv/                # 虚拟环境（自动生成，勿提交）
├── requirements.txt     # 依赖清单
├── src/                 # 核心代码
│   ├── parser.py        # 文献解析：PDF → 段落（带页码/para_id）
│   ├── retriever.py     # 证据检索：BGE 向量 + 中文过滤
│   ├── generator.py     # 综述生成：本地量化大模型
│   ├── match.py         # 句级证据匹配：拆句 + NLI 精筛
│   └── pipeline.py      # 端到端入口（中期检查原型）
├── tests/               # 开发期 spike / 实验脚本（历史参考）
├── docs/                # 团队文档：执行方案 / 分工 / 冲刺计划
├── data/papers/         # 测试用 PDF 文献
├── models/              # 下载的 BGE 等模型权重
├── results/             # 实验记录、输出结果
└── README.md
```


## 环境要求
- Python 3.12
- llama.cpp（CPU 版即可跑通，GPU 版可选）
- Qwen2.5-7B-Instruct Q4_K_M GGUF 模型
- BGE `BAAI/bge-small-zh-v1.5`（本地路径 `models/bge-small-zh-v1.5`，经 ModelScope 下载）
- `transformers==4.46.3`、`sentence-transformers==3.2.1`（**版本组合已锁定，升级会破坏 BGE 和 NLI 的兼容性**）
- NLI：`Fengshenbang/Erlangshen-RoBERTa-110M-NLI`，本地路径
  `models/Fengshenbang--Erlangshen-RoBERTa-110M-NLI/snapshots/master`
  - 标签顺序：`["contradiction", "neutral", "entailment"]`（已实测确认）
  - 加载方式：`BertTokenizer` + `BertForSequenceClassification`（**不要用 `AutoTokenizer`**，会触发 DacConfig 的 KeyError）
## 安装
pip install -r requirements.txt

## 模型下载
- BGE 向量模型：通过 ModelScope 下载 `BAAI/bge-small-zh-v1.5` 到 `models/bge-small-zh-v1.5`
- LLM：下载 Qwen2.5-7B-Instruct Q4_K_M GGUF 到 `D:\llama`（或任意路径）

## 运行
1. 启动 llama-server：
   .\llama-server.exe -m qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -ngl 0 --port 8080
2. 运行端到端原型（中期检查 demo）：
   python src/pipeline.py

### 成员 B 界面（可离线验收）

```bash
python3.12 -m venv .venv-ui
source .venv-ui/bin/activate
python -m pip install --index-url https://pypi.org/simple -r requirements-dev.txt
python src/app.py
```

打开 `http://127.0.0.1:7860`。无需模型即可加载明确标记的演示样例，或导入已有 pipeline JSON，查看逐句证据并下载 Markdown / CSV / JSON / HTML 报告。

真实 PDF 生成需要原有本地模型环境，可通过 `--backend-python /绝对路径/核心环境/bin/python` 选择。界面环境不替代模型环境；缺少模型、PDF 或 llama-server 时会明确报错，不生成假结果。

详见 [成员 B 开发与使用说明](docs/成员B开发说明.md) 和 [成员 B 验收记录](docs/成员B验收记录.md)。


## 当前进度

- [x] 本地大模型推理跑通（llama.cpp + Qwen2.5-7B）
- [x] BGE 向量检索跑通
- [x] PDF 解析 + 清洗（108 段，带页码和 para_id）
- [x] 模块化拆分（parser / retriever / generator / test_pipeline）
- [x] NLI 环境跑通（Erlangshen-110M，标签顺序确认）
- [x] NLI 真实场景测试（3 对，中文证据可用、跨语言证据不可靠）
- [x] **PDF 解析合并逻辑修复**（句子不再被切成碎片，段落语义完整）
- [x] 句级证据匹配接入 pipeline（端到端原型跑通）
- [ ] 人工标注集
- [x] 溯源展示 + 报告导出（离线与 HTTP 验收通过，真实模型联调待做）
- [ ] BM25 混合检索
- [x] Gradio 界面（本地运行；浏览器视觉验收与截图待补）
- [ ] 评价指标

## 已知问题

1. ✅ 英文段落混入检索结果（已解决）：检索侧已加中文占比过滤，阈值 0.3。
2. **NLI 跨语言不可靠**：Erlangshen 是纯中文 NLI 模型，对"英文证据 + 中文生成句"判断不准（实测 S2 语义高度对应却判 neutral）。
3. **PDF 解析警告**：`startxref on same line as offset`、`Multiple definitions in dictionary`，不影响提取。

## 已确定的决策（不得推翻）

1. 项目定位为工程流程整合，不追求底层算法创新
2. 句级匹配只提供证据支持与人工核查依据，不声称证明事实正确
3. PDF 解析只面向文本型 PDF，不处理扫描件 OCR
4. BGE 和 NLI 模型走 ModelScope 下载，代码指向本地路径
5. llama.cpp 暂用 CPU 版，GPU 版暂缓
6. git 提交不 `git add .`，不提交模型和虚拟环境
7. `transformers==4.46.3` 版本锁定，不升级

