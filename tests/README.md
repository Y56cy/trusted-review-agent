# tests/

本目录根部的旧脚本是开发期 spike / 实验，**不是正式测试套件**，仅供历史参考。部分脚本在导入时就访问模型或本地服务，不要对根目录做无差别测试收集。

## 成员 B 正式离线测试

```bash
python -m pytest -q
ruff check src/app.py tests/unit tests/integration/ui_smoke.py
ruff format --check src/app.py tests/unit tests/integration/ui_smoke.py
python tests/integration/ui_smoke.py
```

使用 `.venv-ui` 或已安装 `requirements-dev.txt` 的 Python。pytest 默认只收集 `tests/unit`。HTTP smoke 会启动临时本机 Gradio 服务，并在结束后关闭，不调用真实模型。

- `unit/test_app.py`：结果校验、HTML/CSV 安全、导出、并发隔离与 pipeline 子进程适配。
- `unit/test_app_multipdf.py`：多篇/旧格式契约、同名和 Unicode 冲突、上传限制、来源保留与四种导出安全。
- `unit/test_app_ui.py`：Gradio 组件、离线回调、演示标记和失败清空。
- `unit/test_app_environment.py`：所选核心 Python 自检、混合检索依赖、锁定版本、CLI JSON / 退出码、失败及超时。
- `integration/ui_smoke.py`：真实 HTTP 请求、后端自检、下载、多来源及旧格式 JSON 回导、五文件上传到缺模型后端、同名文件拦截、清空和重复操作。五份上传文件仅含合成 PDF 标识，不是真实解析/模型验证。

这些测试不等于 BGE/NLI/LLM 联调或正式效果评价。

## 历史脚本

- `test_pipeline.py` — 早期的端到端 demo（解析→检索→生成），已被 `src/pipeline.py` 取代
- `test_nil_real.py` — NLI 可行性验证（Erlangshen 中文证据可用、跨语言不可靠）
- `test_nli.py` — NLI 最早的环境测试
- `test_pdf.py` / `test_pdf_clean.py` — PDF 解析与清洗的早期实验
- `test_rag.py` / `test_retrieve.py` / `test_pdf_rag.py` — 模块化之前的检索/生成实验
- `evidence.py` — 打印指定 para_id 段落的小工具

> **正式入口是 `src/pipeline.py`**（中期检查原型），不要在这里改老脚本。
