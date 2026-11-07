# 成员 B：展示与评测开发

## 边界

成员 B 消费 `pipeline.run_review(query, pdf_path, top_k=3, nli_threshold=0.7)` 的既有结果，不改 `parser.py`、`retriever.py`、`generator.py`、`pipeline.py`、`match.py` 或模型接口。NLI 只提供证据支持与人工核查依据，不证明事实正确。

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

原有 `requirements.txt` 保持不变，包括其 UTF-16LE 编码和模型版本。安装该文件失败时请由核心链路维护人确认编码及版本，不在 B 的变更中擅自升级或替换。

## 暂缓事项

- 真实模型端到端验收：仓库未提供 PDF、BGE/NLI 权重和已运行的 llama-server。NLI 路径仍在 `match.py` 中写死为队长的 Windows 路径，需要维护人配置。
- 正式评测跑分：指标公式与人工标注格式由队长确定，目前 `match.compute_metrics` 仍抛出 `NotImplementedError`。不自行发明三项指标公式或伪造标注与分数。
- 原 PDF 页面内的矩形高亮：既有接口没有坐标信息；B 的溯源展示高亮返回的原文片段，不声称支持 PDF 原页坐标定位。
