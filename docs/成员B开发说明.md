# 成员 B：展示与评测开发

## 边界

成员 B 消费 `pipeline.run_review(query, pdf_paths, top_k=3, nli_threshold=0.7)` 的多篇结果（对齐上游 `1d37b0f`），不改 `parser.py`、`retriever.py`、`generator.py`、`pipeline.py`、`match.py` 或模型接口。NLI 只提供证据支持与人工核查依据，不证明事实正确。

## 独立开发环境

使用 Python 3.12，单独创建虚拟环境，避免影响队长和成员 A 已锁定的模型依赖：

```bash
python3.12 -m venv .venv-ui
source .venv-ui/bin/activate
python -m pip install --index-url https://pypi.org/simple -r requirements-dev.txt
python -m pip check
```

Windows PowerShell：

```powershell
py -3.12 -m venv .venv-ui
.\.venv-ui\Scripts\Activate.ps1
python -m pip install --index-url https://pypi.org/simple -r requirements-dev.txt
python -m pip check
```

`requirements-ui.txt` 用于只运行界面；`requirements-dev.txt` 额外安装 pytest、Ruff。`requirements-ui.lock.txt` 是本次 Python 3.12 验证环境的完整解析版本，可用 `pip install -r requirements-ui.lock.txt` 重建开发环境。模型权重、PDF、虚拟环境和临时报告不提交。

选择 Gradio 5.50.0，并限制 `huggingface-hub<1.0`，避免与锁定的 `transformers==4.46.3` 冲突。Gradio 6.31.0 需要 `huggingface-hub>=1.16.0`，不直接混入现有模型环境。界面默认仅监听本机，不启用公共分享链接。

截至主分支 `6f6723e`，`requirements.txt` 已由维护人统一为 UTF-8，并包含检索/UI 依赖及 fsspec 上限。本轮 B 的开发不修改该文件或模型版本；UI 独立环境仍只安装 `requirements-dev.txt`，不会自动安装模型依赖。

## 暂缓事项

- 真实模型端到端验收：仓库未提供 PDF、BGE/NLI 权重和已运行的 llama-server。NLI 路径仍在 `match.py` 中写死为队长的 Windows 路径，需要维护人配置。
- 正式评测跑分：指标公式与人工标注格式由队长确定，目前 `match.compute_metrics` 仍抛出 `NotImplementedError`。不自行发明三项指标公式或伪造标注与分数。
- 原 PDF 页面内的矩形高亮：既有接口没有坐标信息；B 的溯源展示高亮返回的原文片段，不声称支持 PDF 原页坐标定位。

## 已实现内容

- `src/app.py`：独立 Gradio 入口，消费既有 pipeline 的返回值，不修改共享核心模块。
- 上传 1–10 篇文本型 PDF，按选择顺序传入 `pdf_paths`；每篇 ≤50 MiB，总计 ≤200 MiB。中期演示建议 3–5 篇。文献名（文件 stem）重复、仅大小写不同或 Unicode 组合等价会在运行模型前拒绝，避免 `(source, para_id)` 冲突；请自行重命名后重新上传，不在后台悄悄改名。
- 输入研究主题、选择 `top_k` 和 NLI 阈值。单个字符串/Path 的 Python 调用仍可用，但传给最新 pipeline 的键统一为 `pdf_paths` 列表。
- 逐句溯源区域左侧为可点选的综述句子，右侧为证据。点句子只显示该句的证据组；点某条证据链接定位并高亮片段，“查看全部证据”恢复全部组。无证据的句子也有对应提醒，不显示上一句的证据。窄屏变为上下排列，键盘可聚焦链接。
- 每条证据展示 `source` 文献名、页码、段号、完整原文、NLI 标签与模型概率。不同文献相同段号分别保留，不按段号去重或排序。
- 无模型离线模式：导入原始 pipeline JSON 或本界面导出的 JSON；导入结果统一标记“来源未核验”。内置演示必须主动点击，所有演示报告保留演示标记。
- 每次运行生成独立 Markdown、CSV、JSON、HTML 文件。Markdown/HTML 包含综述与证据对照表，CSV 便于逐行人工核查，JSON 便于交接后续跑分。导出只保存现有结果，不创造证据、标注或评价分数。
- HTML 转义、CSV 公式中和、严格数据校验、文件大小限制、每次导出独立目录、错误后清空旧结果。
- 本地 pipeline 在独立子进程运行，工作目录固定为 `src`，默认 600 秒超时；禁止 Hugging Face 隐式联网下载模型。界面和既有核心环境可以使用不同的 Python。
- 真实生成面板新增环境自检，检查与生成相同的核心 Python；显示所检查的解释器、缺失依赖、锁定版本差异、本地模型目录与 llama-server 端口。自检不运行模型，也不覆盖当前综述和下载文件。
- 展示、导出和 JSON 回导保留 pipeline 返回的证据顺序，不按 BGE `score` 或段号重排。混合检索的排名由成员 A 返回的列表次序表达。

## 启动与使用

```bash
python src/app.py
# 端口冲突时：
python src/app.py --port 7861
# 选择另一个已有模型环境，使用绝对路径：
python src/app.py --backend-python /absolute/path/to/core-venv/bin/python
```

Windows 示例：

```powershell
python src/app.py --backend-python 'D:\dachuang\trusted-review-agent\venv\Scripts\python.exe'
```

打开 `http://127.0.0.1:7860`，按以下任一路径使用：

1. **离线演示**：点击“加载演示样例（非真实结果）”，查看有证据和无证据两种句子，再点击证据链接，最后下载报告。样例概率也是手工构造，不可用于答辩的真实效果数据。
2. **已有结果展示**：在“导入已有结果”中上传 JSON，然后点击“导入并展示”。原始格式必须包含 `query`、`review`、`sentences`；证据字段沿用《分工文档》。`page` 和 `para_id` 必须为正整数，概率须为 0–1 的有限数值。`source` 若存在必须是字符串；旧结果缺失时规范化为 `""`，显示“文献来源未知”，不从上传 JSON 文件名或报告 `source_name` 猜测文献。本界面导出的 `schema_version=1` envelope 也可回导。
3. **真实生成**：在核心环境和 llama-server 已准备好后上传 1–10 篇文本型 PDF、输入问题、点击生成。界面没有后台假数据回退，前置条件缺失会显示明确错误。

可直接在界面的“真实生成环境自检”中点击检查，或在命令行指定**核心环境的 Python**：

```bash
# 从 UI 环境检查另一套核心环境（与启动 UI 的 --backend-python 相同）：
python src/app.py --check --backend-python /absolute/path/to/core-venv/bin/python

# 不指定 --backend-python 时检查当前执行它的 Python：
/path/to/core-venv/bin/python src/app.py --check

# 保存机器可读的阻塞清单，便于交接；报告不是评测结果：
python src/app.py --check --backend-python /absolute/path/to/core-venv/bin/python --json
```

自检不加载模型，只检查 Python 3.12、核心与混合检索依赖是否存在、`transformers==4.46.3` / `sentence-transformers==3.2.1` 两个锁定版本、本地权重目录和 llama-server 端口。包括新增的 `numpy`、`jieba`、`rank_bm25`，但不检查每个传递依赖是否能导入，也不校验权重完整性或服务响应内容。

`--check --backend-python` 现在会实际运行所指定的 Python，工作目录和离线环境变量与生成任务相同。子进程自检最多等待 15 秒；不会下载或安装缺失依赖。命令退出码：`0` 为基础检查通过，`1` 为存在缺项，`2` 为自检未能完成或命令用法错误。`--json` 仅与 `--check` 一起使用，自检启动失败时输出 `error`，而非伪造成功报告。

**即使自检通过也不代表模型可加载或生成正确，仍须真实 PDF 联调。** UI 自检单独显示，不替换之前的综述或下载；旧报告仍属于原运行。自检和生成共用串行队列，正在生成时需要等待，不是中断按钮。

## 多文献报告与兼容性

- JSON 仍使用可回导的 `schema_version=1`；每条证据新增/保留 `source`，缺失值保留空串。不会把“未知来源”的展示占位词写成真实文献名。
- Markdown 的逐句表格加入“文献名”列；CSV 在原有九列之后追加“文献名”，原“来源”列仍代表整份报告的来源说明。两者不能混用。
- HTML 报告包含相同的左右对照与逐句筛选，离线可打开。所有格式保留证据返回顺序；HTML/Markdown 转义、CSV 公式中和也覆盖文献名。
- 句子筛选只用固定数字锚点（界面每次加载附加独立前缀）和 CSS `:has()`，没有注入脚本。浏览器不支持 `:has()` 时降级为全部证据显示和锚点跳转。打印 HTML 会展开全部证据。界面清空/重新加载后不会复用上一份结果的锚点；旧浏览器历史不会选中新结果的同号句子。
- 演示样例为两篇手工文献，刻意使用相同 `para_id=1` 验证文献区分，另有无证据句；不是两篇真实论文的运行结果。

## 下载与运行限制

- 文件大小：PDF 上限 50 MiB，JSON 上限 5 MiB。每次最多 10 篇，总计 200 MiB。不处理扫描件 OCR 或 PDF 坐标高亮。多篇生成由上游 pipeline 负责，UI 仅适配上传与结果。
- 下载文件写入临时目录，每次请求隔离；请及时下载保留。进程退出后不保证旧链接仍可用。界面缓存定期清理，源报告临时目录在正常退出时清理。
- 操作共用串行队列，避免多次模型任务争抢资源。正在执行时，后续“清空/导入/演示”会排队，**不是中断按钮**。成功或失败返回前会先清空旧结果；任务最多等待 10 分钟。
- 默认只监听 `127.0.0.1`，`share=False`，不启用公开链接。当前是单机可信用户开发原型，未做面向互联网的部署或安全审计。
- UI 只报告句子和证据数量，不计算“文本忠实度”“引文一致性”或事实准确率。
- 跨语言 NLI 的原有限制仍然存在；导入文件可能来自任意来源，不因可展示而变成经过验证的数据。

## 后续交接清单

1. 队长配置可在当前机器加载的 `_NLI_PATH`，提供本地 BGE/NLI 权重和可运行的核心 Python 环境。
2. 启动 llama-server，并由成员 A / 队长提供授权使用的真实文本 PDF。
3. 在浏览器完成真实生成、证据点击、高亮、下载和重复操作验收，保存标明真实/演示的截图。
4. 队长确定指标公式与标注格式，全员完成标注；成员 B 再跑正式评测并补评测报告。
5. 全部真实验收通过后，再由队长更新《分工文档》《执行方案》的阶段完成状态。

## 中期演示准备

见 [成员 B 多篇联调与录屏清单](成员B多篇联调与录屏清单.md)。已准备可执行验收步骤与录屏顺序，不代表已完成真实 3–5 篇论文生成或正式效果评测。
