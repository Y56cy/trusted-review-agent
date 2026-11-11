# 成员 B 验收记录

## 2026-11-11 本轮结论

基于上游 `1d37b0f` 多篇接口，承接 2026-11-07 的环境自检改动，完成多 PDF 上传、`pdf_paths` 适配、`source` 保留、左右逐句对照、四类报告与旧格式兼容。独立复核没有发现阻断性的代码问题。

**166 项离线单元测试及真实 Gradio HTTP 流程通过；浏览器视觉/点击验收、真实模型端到端运行、人工标注和正式效果跑分未完成。** 五文件 HTTP 上传使用仅含 PDF 标识的合成文件，实际模型前置条件缺失时按预期失败，不是五篇论文成功生成。

本轮相对上游只修改 B 的 `app.py`、相关测试、README 与 B 的说明/验收/联调清单。未修改 `parser.py`、`retriever.py`、`generator.py`、`pipeline.py`、`match.py`、模型依赖或队长维护的三份进度文档。原预检分支保留。

## 当前检查记录

环境：Python 3.12.14，独立 `.venv-ui`，Gradio 5.50.0。

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| `python -m pip check` | 通过 | UI 环境依赖无冲突 |
| `python -m pytest -q` | 166 passed | `tests/unit`，不收集历史模型实验 |
| `ruff check src/app.py tests/unit tests/integration/ui_smoke.py` | 通过 | B 的 UI 与回归测试 |
| `ruff format --check src/app.py tests/unit tests/integration/ui_smoke.py` | 通过 | 格式检查 |
| `python -m py_compile src/app.py` | 通过 | 语法检查 |
| `python tests/integration/ui_smoke.py` | 通过 | 实际本地 HTTP 服务、上传与下载 |
| `python src/app.py --check --backend-python <UI Python> --json` | 按预期退出 1 | 列出当前核心依赖/模型/服务缺项 |
| `git diff --check` | 通过 | 空白与补丁检查 |
| 上游 `tests/retrieval` | 未能收集 | UI 环境缺少 jieba；不计为通过 |

两条 Gradio 6.0 API 弃用预告不是测试失败，不为消除预告升级已选兼容版本。

## 本轮新增覆盖

- 一篇或多篇路径规范化；五篇请求顺序保留并以 `pdf_paths` 传给 worker；实际 `pipeline.py` 在依赖桩下完成五篇编排，验证 source 贯穿解析、索引、提示词与匹配。依赖桩测试不代表模型效果。
- 最多 10 篇、单篇 50 MiB、合计 200 MiB；每个文件的扩展名/标识校验；相同 stem、大小写冲突及 Unicode NFC 等价冲突在运行模型前拒绝。
- 不同文献相同 `para_id` 不合并；证据列表保持上游顺序，不根据分数、文献名或段号排序。
- `source` 原样保留到校验、显示及 JSON/CSV/Markdown/HTML；拒绝非文本 source；旧结果缺 source 使用空串和未知来源提示，绝不从报告元数据猜测文献名。
- 文献名中的 HTML/Markdown 特殊字符被转义；CSV 公式注入被中和；JSON 保留原始数据并可回导。
- 句子链接和证据链接均有独立目标，无证据句包含自己的提醒；左右区域及窄屏样式存在。每次界面加载使用全新锚点前缀，旧历史/旧选句不会匹配新结果。
- 沿用环境自检、超时、输出清空、并发导出隔离、来源未核验和手工演示标记等原有回归。

HTTP smoke 实际执行：两文献手工演示 → 四格式下载 → 多来源 JSON 回导 → 缺 source 旧 JSON 回导 → 自检 → 清空/坏输入 → 重复文件拒绝 → 五文件上传并如实报告模型缺项 → 重新加载，校验新旧锚点不重叠。

## 未完成与交接

### 浏览器视觉与交互

本轮 CUA 浏览器访问临时 `127.0.0.1:7877` 返回 `net::ERR_CONNECTION_REFUSED`，未建立到测试服务的连接，临时服务已关闭。没有截图，也未声称浏览器真实点击、滚动、CSS 筛选、窄屏和后退/前进已经验收。未通过公开分享或修改网络设置绕过限制。

HTML 结构及 HTTP 回归不能替代视觉检查。请在可访问本机服务的开发机按 [多篇联调与录屏清单](成员B多篇联调与录屏清单.md) 验收。CSS `:has()` 不支持时降级为完整证据列表和锚点跳转。

### 真实模型链路

当前 UI 环境缺少 pypdf、torch、transformers、sentence_transformers、sklearn、jieba、rank_bm25；本地 BGE/NLI 权重缺失，8080 无 llama-server，仓库没有真实 PDF。上游文档的另一台机器已通过环境检查不代表当前机器具备这些条件。

`--backend-python` 可选择准备好的核心环境，但不会替代模型/文献准备。未下载大模型、伪造论文数据或真实生成结果。基础自检只检查发现依赖、锁定版本、目录及端口，不校验模型完整性、服务身份或生成正确性。

### 正式效果评价

指标公式、人工标注与 `compute_metrics` 尚未就绪，本轮不自行定义分数。演示的 NLI 概率也是手工构造，不能用于效果报告；JSON/CSV 仅供后续人工核查和跑分交接。
