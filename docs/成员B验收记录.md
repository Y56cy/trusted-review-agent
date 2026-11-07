# 成员 B 验收记录

## 验收结论

环境、界面、溯源片段展示、报告导出和离线回归已完成。真实 HTTP 服务流程验收通过。**未完成浏览器视觉验收/运行截图、真实模型端到端验收、人工标注和正式效果跑分。**

本记录只覆盖成员 B 的新增文件。`parser.py`、`retriever.py`、`generator.py`、`pipeline.py`、`match.py` 和原模型依赖文件均未修改。

## 已通过的检查

运行环境：Python 3.12.14，独立 `.venv-ui`，Gradio 5.50.0。完整依赖版本见 `requirements-ui.lock.txt`。

| 检查 | 结果 | 范围 |
| --- | --- | --- |
| `python -m pip check` | 通过 | UI 开发环境依赖无冲突 |
| Gradio 导入 / Blocks 构建 | 通过 | 无模型导入、无模型下载 |
| `python -m pytest -q` | 87 passed | 仅 `tests/unit`，不收集历史模型实验 |
| `ruff check src/app.py tests/unit tests/integration/ui_smoke.py` | 通过 | 新增 B 代码 |
| `ruff format --check src/app.py tests/unit tests/integration/ui_smoke.py` | 通过 | 新增 B 代码 |
| `python -m py_compile src/app.py` | 通过 | 语法检查 |
| `python tests/integration/ui_smoke.py` | 通过 | 实际本地 Gradio HTTP 服务与文件传输 |
| `git diff --check` | 通过 | 空白与补丁检查 |

87 项离线回归覆盖：

- pipeline 结果结构、空结果、非法类型、NaN/Infinity、巨大整数、文件大小和损坏 JSON。
- 所有动态 HTML 文本的转义、内部固定锚点、Markdown 导出、CSV 公式中和。
- 中文原文完整保留、无证据提醒、演示标记、未核验来源、未进行正式评测的声明。
- 16 次并发导出路径隔离、默认临时根目录首次初始化的并发保护。
- app 模块导入时不加载 Gradio/pipeline/torch/transformers。
- 子进程参数、正确工作目录、离线模型开关、超时及错误清空。
- Gradio 事件接线、串行队列、生成/导入失败后不残留旧下载。

HTTP smoke 通过实际客户端请求验证：

1. 启动本机临时端口的 Gradio 服务并读取配置。
2. 运行明确标记的手工演示，下载四种非空报告。
3. 回导导出的 JSON，保留演示标记，并显示来源未核验。
4. 清空结果，验证证据和下载返回为空。
5. 空问题、损坏 JSON 友好失败，无旧结果残留。
6. 重复加载演示，产生独立的导出文件。

这些请求未调用 BGE、NLI 或 LLM。两条 Gradio 警告是面向 6.0 的 API 弃用提示，不是测试失败；不为消除警告而升级已选择的兼容版本。

## 未完成与阻塞

### 浏览器视觉验收与截图

已尝试用本机已安装的 Chromium 启动浏览器，包括获授权的本地测试运行，但环境在创建 Unix socket 时返回 `Operation not permitted`，浏览器退出。没有可用截图，也没有声称实际点击后的视觉高亮或移动端布局已经验收。

当前已验证 HTML 的证据链接与唯一目标 ID 对应，包含高亮 CSS；HTTP 流程通过。后续需在允许启动浏览器的开发机上按开发说明执行人工浏览器验收。

### 真实模型联调

`python src/app.py --check` 在 UI 专用环境中按预期返回失败，具体缺失：

- pypdf、torch、transformers、sentence_transformers、sklearn 核心依赖未安装在 UI 环境。
- 本地 BGE 权重目录不存在。
- `match.py` 的 Windows `_NLI_PATH` 在当前机器不存在。
- `127.0.0.1:8080` 未运行 llama-server。
- 仓库没有真实 PDF 测试文件。

UI 的 `--backend-python` 可以复用已配置好的核心模型环境，但不能替代这些准备。未下载大模型、未修改其他成员接口、未伪造生成内容。

### 正式评测

队长尚未确认三项指标公式和标注格式，`match.compute_metrics` 仍未实现，没有人工标注集。此次不生成任意公式、标注或“已通过”的分数。JSON/CSV 导出可供后续人工核查与正式跑分使用。

## 复验说明

首次把 HTTP smoke 整理成独立脚本时，30 秒启动预算曾超时，未进入任何测试请求；随后单独复跑通过，独立复核也复跑通过。脚本现保留 120 秒启动预算，并在启动失败时打印日志、关闭子进程。该调整只放宽服务冷启动等待，不跳过流程断言，也不把超时计作通过。
