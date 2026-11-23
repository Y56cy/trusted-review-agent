# 基于智能体的可信可溯源科研综述辅助系统
面向高校图书馆科研服务场景的本地化科研综述辅助原型。核心链路：

`文献解析 → 证据检索 → 综述生成 → 句级校验 → 溯源展示 → 效果评价`

## 项目定位（重要）

本项目是**工程流程整合**，不追求底层算法创新。不重新训练大模型，不提出新生成算法，而是把 RAG、本地大模型、文本匹配技术整合成高校图书馆场景下的可追溯工作流。

**句级证据匹配只用于提供"证据支持与人工核查依据"，不声称能自动证明结论为真。** 这是已确定的边界，后续实现不得越过。

## 新成员上手（先看这里）

1. 看 `docs/进展规划.md` —— 项目方向（中期 → 结题）
2. 看 `docs/结题执行评估方案.md` —— 结题阶段要做什么、怎么验收
3. 看 `docs/执行方案.md` —— 阶段划分与功能清单

代码入口是 `src/pipeline.py`，一条命令跑通全链路。团队进度以《结题执行评估方案》的 checkbox 为准。中期过程文档已归档到 `docs/归档/`。

## 目录结构

```
├── venv/                # 虚拟环境（自动生成，勿提交）
├── requirements.txt     # 依赖清单
├── src/                 # 核心代码
│   ├── parser.py        # 文献解析：PDF → 段落（带页码/para_id）
│   ├── retriever.py     # 证据检索：BGE + BM25 / RRF + 中文过滤
│   ├── generator.py     # 综述生成：本地量化大模型
│   ├── match.py         # 句级证据匹配：拆句 + NLI 精筛
│   └── pipeline.py      # 端到端入口（中期检查原型）
├── tests/               # 开发期 spike / 实验脚本（历史参考）
├── docs/                # 团队文档：执行方案 / 分工 / 冲刺计划
├── data/papers/         # 单篇测试用 PDF 文献
├── data/corpus/         # 多篇文献库（本次冲刺新建，RAG 主题 5 篇）
├── models/              # 下载的 BGE 等模型权重
├── results/             # 实验记录、输出结果
└── README.md
```


## 环境要求
- Python 3.12
- llama.cpp（CUDA 版走 GPU 加速；纯 CPU 可用 `-ngl 0` 但慢）
- Qwen2.5-7B-Instruct Q4_K_M GGUF 模型
- BGE `BAAI/bge-small-zh-v1.5`（本地路径 `models/bge-small-zh-v1.5`，经 ModelScope 下载）
- `transformers==4.46.3`、`sentence-transformers==3.2.1`（**版本组合已锁定，升级会破坏 BGE 和 NLI 的兼容性**）
- NLI：`Fengshenbang/Erlangshen-RoBERTa-110M-NLI`，本地路径
  `models/Fengshenbang--Erlangshen-RoBERTa-110M-NLI/snapshots/master`
  - 标签顺序：`["contradiction", "neutral", "entailment"]`（已实测确认）
  - 加载方式：`BertTokenizer` + `BertForSequenceClassification`（**不要用 `AutoTokenizer`**，会触发 DacConfig 的 KeyError）
## 安装
pip install -r requirements.txt
# 该文件已用 -r 引用 requirements-retrieval.txt 和 requirements-ui.txt，一条命令装齐全部依赖。

成员 A 的检索模式、独立测试、示例对照与真实论文补验步骤见 [成员 A 检索开发说明](docs/归档/成员A检索开发说明.md) 和 [成员 A 验收记录](docs/归档/成员A验收记录.md)。示例评测不代表真实论文表现。

## 模型下载
- BGE 向量模型：通过 ModelScope 下载 `BAAI/bge-small-zh-v1.5` 到 `models/bge-small-zh-v1.5`
- LLM：下载 Qwen2.5-7B-Instruct Q4_K_M GGUF（分卷 `00001-of-00002` / `00002-of-00002`，放同一目录）
- llama.cpp：从 [官方 Releases](https://github.com/ggml-org/llama.cpp/releases) 下载 CUDA 版，两个包都要解压到同一目录：
  - `llama-bXXXXX-bin-win-cuda-12.4-x64.zip`（主程序，含 `llama-server.exe`）
  - `cudart-llama-bin-win-cuda-12.4-x64.zip`（CUDA 运行库，缺它启动会报"缺 DLL"）

## 运行
1. 启动 llama-server（GPU 加速）：
   .\llama-server.exe -m <模型完整路径>\qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf -ngl 99 --port 8080
   - `-m` 要用模型文件**完整路径**（只写文件名会报 file not found）。
   - `-ngl 99` = 全层卸载到显卡；日志出现 `offloaded 28/28 layers to GPU` 即生效。纯 CPU 环境改 `-ngl 0`（慢）。
2. 运行端到端原型（中期检查 demo，默认读 `data/corpus/` 多篇语料）：
   python src/pipeline.py
   语料就位后即可演示「多篇文献库 → 检索 → 综述生成 → 逐句溯源」。

### 成员 B 界面（可离线验收）

```bash
python3.12 -m venv .venv-ui
source .venv-ui/bin/activate
python -m pip install --index-url https://pypi.org/simple -r requirements-dev.txt
python src/app.py
```

打开 `http://127.0.0.1:7860`。无需模型即可加载明确标记的演示样例，或导入已有 pipeline JSON，左右对照查看逐句证据并下载 Markdown / CSV / JSON / HTML 报告。支持一次上传 1–10 篇文本型 PDF，文献名贯穿展示与导出；旧结果缺少文献名时明确标记未知。

真实多篇 PDF 生成需要原有本地模型环境，可通过 `--backend-python /绝对路径/核心环境/bin/python` 选择。界面环境不替代模型环境；缺少模型、PDF 或 llama-server 时会明确报错，不生成假结果。

生成前可运行 `python src/app.py --check --backend-python /绝对路径/核心环境/bin/python` 检查环境。加 `--json` 可输出交接用自检报告；自检不加载模型，通过也不代表真实联调已完成。

详见 [成员 B 开发与使用说明](docs/归档/成员B开发说明.md) 和 [成员 B 验收记录](docs/归档/成员B验收记录.md)。


## 当前进度

- [x] 本地大模型推理跑通（llama.cpp + Qwen2.5-7B，GPU 加速）
- [x] BGE 向量检索跑通
- [x] PDF 解析 + 清洗（108 段，带页码和 para_id）
- [x] 模块化拆分（parser / retriever / generator / match）
- [x] NLI 环境跑通（Erlangshen-110M，标签顺序确认）
- [x] 句级证据匹配接入 pipeline（端到端原型跑通）
- [x] BM25 / BGE RRF 混合检索 + 检索评测工具（组员 A）
- [x] 多篇文献库（加 `source` 字段 + 5 篇 RAG 主题文献，443 段）
- [x] 左右对照 UI + 四格式导出（组员 B）
- [x] **生成-校验-重写闭环**（主动防幻觉：无证据句自动改写 / 显式标「未能溯源」）
- [x] **两智能体综述扩写**（可溯源关键句 → 结构化综述）
- [x] **引用标注**（`[N]` 上标 + 参考文献表，贯穿展示与导出）
- [x] 真实文献检索对照评测完成（5 篇、443 段，6 条开发集与 14 条保留集，比较 BGE、BM25 与 RRF）
- [ ] 评价指标 `compute_metrics`（可追溯度，公式待定）
- [ ] PDF 原页坐标高亮（结题迭代项）

## 已知问题

1. ✅ 英文段落混入检索结果（已解决）：检索侧已加中文占比过滤，阈值 0.3。
2. **NLI 跨语言不可靠**：Erlangshen 是纯中文 NLI 模型，对"英文证据 + 中文生成句"判断不准。
3. **综合/总结句难以溯源**：综述里"总结/综合"性质的句子（往往是主题句）天然难锚定单条证据，导致部分查询未能溯源率偏高；已交由综述扩写的引言/过渡/总结承担。
4. **长段落超 BGE 512 token 截断**：段末证据可能漏召回、Recall 偏低，评测时如实说明。
5. **PDF 解析警告**：`startxref on same line as offset`、`Multiple definitions in dictionary`，不影响提取。

## 已确定的决策（不得推翻）

1. 项目定位为工程流程整合，不追求底层算法创新
2. 句级匹配只提供证据支持与人工核查依据，不声称证明事实正确
3. PDF 解析只面向文本型 PDF，不处理扫描件 OCR
4. BGE 和 NLI 模型走 ModelScope 下载，代码指向本地路径
5. llama.cpp 用 CUDA 版走 GPU 加速（`-ngl 99`）；纯 CPU 仅作降级兜底
6. git 提交不 `git add .`，不提交模型和虚拟环境
7. `transformers==4.46.3` 版本锁定，不升级
8. 文献库方向：大模型/RAG 主题，中期演示 5 篇 PDF；文献名用文件名
9. 溯源展示先做「点句 → 右侧显示证据原文 + 文献名/页码」，PDF 原页坐标高亮是结题迭代项
10. 多篇段落唯一标识 = (source, para_id)；para_id 篇内从 1 开始

