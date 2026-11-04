# tests/

开发期的 spike / 实验脚本，**不是正式测试套件**，仅供历史参考。

- `test_pipeline.py` — 早期的端到端 demo（解析→检索→生成），已被 `src/pipeline.py` 取代
- `test_nil_real.py` — NLI 可行性验证（Erlangshen 中文证据可用、跨语言不可靠）
- `test_nli.py` — NLI 最早的环境测试
- `test_pdf.py` / `test_pdf_clean.py` — PDF 解析与清洗的早期实验
- `test_rag.py` / `test_retrieve.py` / `test_pdf_rag.py` — 模块化之前的检索/生成实验
- `evidence.py` — 打印指定 para_id 段落的小工具

> **正式入口是 `src/pipeline.py`**（中期检查原型），不要在这里改老脚本。
