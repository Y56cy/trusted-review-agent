# 成员 A：多文献标注与评测操作说明

## 当前交付范围

本轮在 `feat/multi-source-evaluation` 分支交付，已推送 GitHub，通过 Draft PR 交队长审查。
基线为主分支 `1d37b0f`，不修改 parser、retriever、pipeline、match、generator 或 UI。
多文献身份统一为 `(source, para_id)`，source 为 PDF 文件名去掉扩展名，段号保持篇内编号。
本说明补充并优先于旧《成员A检索开发说明》中的单篇输入与整数标签描述。

## 环境与测试

继续用 Python 3.12.14、既有本地 BGE。安装入口沿用队长最新的 `requirements.txt`，
其中已包括检索和 UI 依赖，fsspec 约束为 `>=2023.5.0,<=2026.6.0`。
本机核心环境的 fsspec 已对齐到 2026.6.0；本轮没有安装或启动 Gradio。

```powershell
venv/Scripts/python.exe -m unittest discover -s tests/retrieval -v
venv/Scripts/python.exe -m pytest tests/unit/test_app.py -q
uv pip check --python venv/Scripts/python.exe
```

新增代码只依赖标准库，解析 PDF 时才需要已有 pypdf，评测时才加载 BGE。
测试中的临时 PDF 和向量替身只验证接口、计分、材料生成，不能当作真实论文评测。

## 第一步：接收并冻结语料

由组长确定主题，全员协作准备目标 5 篇论文，演示最低 3 篇。文件放本地 `data/papers/`，
文件名主体不能重复，不要在标注后重命名。只能接收可提取文字的文本型 PDF。

从仓库根目录运行（把示例文件名替换为真实文件名）：

```powershell
venv/Scripts/python.exe scripts/evaluate_retrieval.py `
  --pdf data/papers/论文甲.pdf data/papers/论文乙.pdf data/papers/论文丙.pdf `
  --prepare-only --output-dir results/retrieval-materials-v1
```

此步骤不需要 queries、不加载 BGE，也不输出相关性分数，生成：

| 文件 | 用途 |
|---|---|
| corpus.json | 后续固定使用的段落快照及清单 |
| corpus-manifest.json | 文献名、文件名、PDF SHA-256、段落数、解析结果 SHA-256 |
| paragraphs.html | 本地查看原文，按文献、页码、段号展示；内容经过 HTML 转义 |
| queries.template.json | 空查询列表、格式示例、语料绑定值、待填写的复核信息 |

每篇抽查开头、中间和末尾，对照原 PDF 核对文字与页码。解析问题交给队长，不手改段号。
已有材料目录不允许重复覆盖；换新版本目录重新准备。语料变化后必须复核旧标签。
快照会核对内容校验值、逐文献段落数与校验值；PDF 哈希属于记录值，原 PDF 仍由团队保存核对。

## 第二步：编写与复核查询

复制 `queries.template.json` 为 `queries.draft.json`，保留 `corpus_sha256`。
它绑定规范化段落快照（不是 PDF 文件或 JSON 文件字节哈希），请不要手改。
填入 queries 列表，目标 20 条：6 条 dev、14 条 heldout。若组长压缩为 10—19 条，
dev 取数量的 30% 向下取整，其余 heldout。先固定划分再运行评测。

格式如下（结构示例，不能作为真实标签直接使用）：

```json
{
  "annotation_status": "draft",
  "annotator": "实际标注人",
  "reviewer": "",
  "review_notes": "待独立复核",
  "corpus_sha256": "从工具模板保留的值",
  "queries": [
    {
      "query_id": "q001",
      "query": "按原文编写的真实问题",
      "split": "dev",
      "relevant_paragraphs": [
        {"source": "论文甲", "para_id": 3},
        {"source": "论文乙", "para_id": 7}
      ]
    }
  ]
}
```

- 问题涵盖定义术语、方法、实验条件、局限、同义改写、跨文献比较。
- 标签根据完整原文编写，不能先看三种检索排名再反推标准答案；不要求强行包含跨篇证据。
- 相关段落应直接回答或提供必要证据，共享关键词不算相关。所有相关段落尽量标全。
- 无法由整库回答的问题另存边界案例，不混入要求至少一个相关段落的 P/R/MRR 数据集。
- 由另一成员看原文独立复核，记录分歧与决定，确认后保存为 `queries.reviewed.json`。
- 复核后把 `annotation_status` 改为 `reviewed`，填写不同的 annotator/reviewer 和非空 review_notes。
  可以逐查询覆盖头部字段；只把实际确认的标签标为 reviewed。
- 复核声明及绑定缺失或不完整时可以输出初步评测，但不能冒充正式验收。
  程序只能检查声明是否完整，不能证明人工复核确实发生。

旧 `relevant_para_ids` 整数列表只在编号全库唯一时自动转换；跨文献重复时明确报歧义。
同一查询不能同时包含两种标签字段；多文献不能缺失 source。

## 第三步：校验、开发集与保留集评测

优先使用冻结的 corpus.json，不重新解析 PDF：

```powershell
venv/Scripts/python.exe scripts/evaluate_retrieval.py `
  --paragraphs results/retrieval-materials-v1/corpus.json `
  --queries results/retrieval-materials-v1/queries.draft.json --validate-only

venv/Scripts/python.exe scripts/evaluate_retrieval.py `
  --paragraphs results/retrieval-materials-v1/corpus.json `
  --queries results/retrieval-materials-v1/queries.draft.json `
  --split dev --output-dir results/retrieval-real-dev-v1

venv/Scripts/python.exe scripts/evaluate_retrieval.py `
  --paragraphs results/retrieval-materials-v1/corpus.json `
  --queries results/retrieval-materials-v1/queries.reviewed.json `
  --split heldout --output-dir results/retrieval-real-heldout-v1
```

`--pdf` 仍可直接评测一篇或多篇；`--paragraphs` 与它互斥。
默认 k=3、中文占比=0.3、BM25/RRF 参数保持不变。只在开发集检查错误或讨论必要调整，
参数冻结后才运行保留集，修 bug 重跑时记下原因、保存新目录，不覆盖旧报告。
工具会拒绝向已有 report.json/report.md 的目录重新写报告。

P/R/MRR 按 `(source, para_id)` 计分；被语言过滤排除的已标注相关段落仍保留在分母中。
报告中 score 仍为余弦相似度，不得据此重排混合结果。

## 报告与失败案例

report.json 与 report.md 包含文献清单、语料/模型/代码/依赖身份、复核声明、集合划分、
逐查询五字段排名和命中、三模式宏平均指标、耗时及案例分类。
`case_analysis` 分类为 improved、unchanged、declined、missed、no_results；
同一查询某指标升某指标降可同时出现，不能只挑改善案例。

至少人工说明 3 个代表性案例，优先覆盖改善、无改善或下降、漏召回；没有某类就如实写不存在。
工具只列事实和排名，不自动猜原因。对照原文逐项检查解析质量、分词、语义召回、语言过滤、
候选截取、标注争议，记录核查证据；不可仅凭平均分判断原因。
Precision 返回不足 k 条仍除以 k，Recall 除以全部相关段落数，MRR 是首个相关位置的倒数。
这些指标评价检索，不评价综述事实正确性或 NLI 判定准确性。

## D6 联调与当前待办

使用相同冻结语料向队长提供查询和返回证据，核对 source、页码和段号全程一致。
完整生成与界面由队长/B 的模型环境验收；本轮不修改他们的代码。

- [x] 多文献评测适配、准备材料、校验与自动化测试。
- [ ] 组长确定主题，全员协作准备 3—5 篇真实 PDF。
- [ ] 真实语料抽查与冻结；20 条问题初稿和另一成员独立复核。
- [ ] 正式保留集对照评测、至少 3 个代表案例人工分析。
- [ ] 完整生成链路联调。

批量原文、PDF、真实标注、模型和运行报告先全部保留在 Git 已忽略的本地目录。
Draft PR 仅审查工具适配，不代表真实评测完成；由队长审查合并，不提前更新团队任务完成状态。
