# 成员 A：混合检索与独立评测

## 交付边界与快速理解

BM25 依靠词频与稀有词匹配明确术语；BGE 用语义向量寻找措辞不同的段落。
RRF 把两路的排名融合，不直接相加两种原始分数。第一版双通道权重各 0.5、常数 60，
每个通道取 `min(符合语言过滤的段落数, max(20, top_k))` 个候选。
BM25 使用 `k1=1.5, b=0.75`，只收正分候选；没有正分候选时混合模式回退到向量排序。
IDF 使用 `ln(1 + (N - df + 0.5) / (df + 0.5))`，避免库默认公式在一两段语料或常见词上
产生零/负分、错误丢弃准确匹配。关键词通道同分只按段落编号排序，不混入向量排名。
这保证了处理流程可解释，但不保证所有语料上的检索指标都会提高。

只修改检索层；解析、生成、NLI、UI 保持原状。本交付不是综述忠实度或事实正确性评价。

## 环境与模型

以下 PowerShell 命令从仓库根目录运行，依赖清单按顺序安装：

```powershell
uv venv --python 3.12.14 venv
uv pip install --python venv/Scripts/python.exe -r requirements.txt -r requirements-retrieval.txt
```

先确认现有锁定版本可用。安装失败时保留错误记录，不擅自更换 torch、transformers、
sentence-transformers 等锁定依赖。逻辑测试和真实模型推理是两项独立验收。

本模块默认只读取 `models/bge-small-zh-v1.5` 中的本地模型，不自动联网下载。
目录不存在会提示 ModelScope 下载方式；不需要 Qwen、NLI 或 llama-server。

```powershell
venv/Scripts/python.exe -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-zh-v1.5', local_dir='models/bge-small-zh-v1.5')"
```

默认模型路径相对模块所在仓库解析；显式 `model_path` 优先，显式相对路径按调用时工作目录解析。
权重、虚拟环境和原始 PDF 不提交到 Git。

## 接口与排序

```python
from retriever import Retriever

retriever = Retriever()  # 或 Retriever(model_path="本地模型目录")
retriever.build_index(paragraphs)  # [{"text": ..., "page": 1, "para_id": 1}]
results = retriever.search(query, top_k=3, min_chinese_ratio=0.3)
baseline = retriever.search(query, method="vector")
lexical = retriever.search(query, method="bm25")
```

三个原有位置参数仍可用，新增 `method` 是仅限关键字的可选参数，默认 `hybrid`。
仍只返回 `text / page / para_id / score` 四字段。**score 始终是 BGE 余弦相似度；
hybrid 的列表次序才是融合排名，不得再按 score 重排。** pipeline 与 UI 无须改调用。

- 中文分词使用 jieba 默认词典、`HMM=False`；英文小写，去掉空白和纯标点，保留中文单字、英文、数字片段。
- 索引缓存段落分词和中文占比。中文字符占非空白字符的比例默认须达到 0.3，过滤先于候选截取。
- BM25 索引使用全部输入段落的词频统计；过滤只限制候选，三种模式的可选语料完全相同。
- RRF 同分先按向量相似度降序，再按正整数 `para_id` 升序。段落编号必须唯一。
- 未建索引、空语料、非法编号或参数会报错；建好索引后的空白查询返回空列表。
- 中文候选不足时按实际数量返回。BM25 可能因无正分命中返回空列表。
- 建索引成功后原子替换旧语料与双索引，失败不会留下半更新状态。

为保持已有 API，BM25 模式也计算余弦相似度作为返回 score，因此仍要求本地 BGE。
首版为小语料内存检索，不持久化索引，每次 `run_review` 仍按原逻辑重新建索引。

## 独立测试与示例评测

```powershell
venv/Scripts/python.exe -m unittest discover -s tests/retrieval -p 'test_*.py' -v
venv/Scripts/python.exe scripts/evaluate_retrieval.py --paragraphs data/retrieval/paragraphs.example.json --queries data/retrieval/queries.example.json --validate-only
venv/Scripts/python.exe scripts/evaluate_retrieval.py --paragraphs data/retrieval/paragraphs.example.json --queries data/retrieval/queries.example.json --output-dir results/retrieval
```

不要对整个 `tests/` 直接做 unittest/pytest 自动发现：那里保留了会加载模型的历史实验脚本。
现有成员 B 的 pytest 配置继续只发现 `tests/unit`，本模块测试使用上述显式目录。

示例包含 20 个原创段落、12 条查询（4 dev / 8 heldout），不是从真实论文提取。
相关段落标签在运行检索之前编写，尚未独立人工复核，不能写成真实人工标注数据集。
第 20 段是中文引导语加英文参考文献的低中文占比样例，第 18、19 段提供词面干扰。
固定参数后运行一次保留集，允许并如实报告混合模式持平或下降，不根据保留集反复调参。
调试阶段可在评测命令后加 `--split dev`，完全不执行保留集查询；最终用 `--split heldout`
单独验收，或 `--split all`（默认）生成完整对照。所有输入标签仍在模型加载前完整校验。

报告生成 `report.md` 与 `report.json`，包含分组指标、每条查询排名、命中、运行配置、
源文件及解析结果校验值、模型文件校验值、依赖版本、模型加载/索引/查询耗时。
报告同时记录检索与评测代码校验值。自定义 `--model-path` 显示为 `local:目录名`，
不会伪称默认 BGE；具体文件身份以模型文件校验值为准。
查询按固定模式顺序执行，耗时包含缓存与预热影响，只是本机观察值，不构成性能对比结论。
报告目录为运行产物，不提交；需要分享实验结果时检查数据后另行提供。

三项指标均按查询取平均：

| 指标 | 含义 | 示例：相关编号 1、3，返回 2、3，k=3 |
|---|---|---|
| Precision@k | 前 k 位相关命中数 / k，返回不足也除以 k | 1/3 |
| Recall@k | 前 k 位相关命中数 / 全部标注相关数 | 1/2 |
| MRR@k | 首个相关排名的倒数，未命中为 0 | 1/2 |

## 真实论文补验步骤

1. 取得队长同一份文本型 PDF，记录 SHA-256，用既有 `parse_pdf` 提取段落。
2. 将解析结果保存为段落 JSON 供标注者查看。按同一编号体系标注至少 20 个真实问题，
   记录相关段落集合与 dev/heldout 划分；由另一成员复核，记录分歧处理。
3. 查询文件使用列表或 `{"queries": [...]}` 外壳；每条有唯一非空 `query_id`、非空 `query`、
   非空正整数列表 `relevant_para_ids`、`split`（`dev` 或 `heldout`）。
   语料也可用列表或 `{"paragraphs": [...]}` 外壳。外壳可注明 `dataset_type` 和标注来源。
4. 三种模式使用相同 PDF、查询、k、语言过滤阈值。只在开发集选择参数；首次交付参数固定，不自动调参。

```powershell
venv/Scripts/python.exe scripts/evaluate_retrieval.py --pdf data/papers/paper.pdf --queries data/retrieval/queries.real.json --model-path models/bge-small-zh-v1.5 --output-dir results/retrieval-real
```

5. 核对保留集的平均指标及逐查询失败案例。对过滤掉的有效相关段落同样算漏召回，不偷偷删除标签。
6. 完成上述数据与独立复核前，不把“真实检索效果已验证”标记完成；NLI 与综述指标属于队长任务。

## 交接时应能解释的三件事

- BM25 擅长找准确术语，BGE 擅长找语义相近的说法；混合检索的价值必须由对照实验验证。
- RRF 利用两个排名的倒数，解决原始分数量纲不同的问题；score 为兼容旧接口仍保留向量含义。
- 高 Recall 表示少漏相关段落，高 Precision 表示返回更集中，高 MRR 表示更早看到第一个相关结果；
  三者都不能证明语言模型生成的结论正确。
