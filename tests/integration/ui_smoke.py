"""Exercise the real local Gradio HTTP API without models or browser automation.

Run with: python tests/integration/ui_smoke.py
This explicitly synthetic smoke test is not a real model or evaluation run.
"""

from __future__ import annotations

import hashlib
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

from gradio_client import Client, handle_file

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory(prefix="review-smoke-") as directory:
        log_path = Path(directory) / "server.log"
        with log_path.open("w+") as log:
            server = subprocess.Popen(
                [sys.executable, "-u", "src/app.py", "--port", str(port)],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    try:
                        with urlopen(f"{url}/config", timeout=1) as response:
                            if response.status == 200:
                                break
                    except OSError:
                        if server.poll() is not None:
                            raise RuntimeError(log_path.read_text("utf-8")) from None
                        time.sleep(0.2)
                else:
                    raise RuntimeError("Gradio 未在 120 秒内启动：" + log_path.read_text("utf-8"))
                client = Client(url, verbose=False)
                result = client.predict(api_name="/show_demo")
                assert "手工构造" in result[0]
                assert len(result[3]) == 4
                assert 'class="trace-columns"' in result[2]
                assert re.search(r'href="#run-[0-9a-f]+-sentence-2"', result[2])
                for filename in result[3]:
                    assert Path(filename).is_file() and Path(filename).stat().st_size > 0
                diagnostics = client.predict(api_name="/check_environment")
                assert "检查的 Python：" in diagnostics
                assert "不加载模型" in diagnostics and "仍须使用真实 PDF" in diagnostics
                envelope = json.loads(Path(result[3][2]).read_text("utf-8"))
                assert envelope["metadata"]["is_demo"] is True
                sources = [e["source"] for e in envelope["result"]["sentences"][0]["evidences"]]
                assert len(set(sources)) == 2
                assert all(
                    source in imported_html
                    for source in sources
                    for imported_html in [result[2], Path(result[3][3]).read_text("utf-8")]
                )
                imported = client.predict(handle_file(result[3][2]), api_name="/import_result")
                assert "来源未核验" in imported[0] and "手工构造" in imported[0]
                assert imported[1] == result[1]
                assert all(source in imported[2] for source in sources)
                paragraphs = [
                    {k: e[k] for k in ("source", "page", "para_id", "text")}
                    for e in envelope["result"]["sentences"][0]["evidences"]
                ]

                def checksum(value):
                    return hashlib.sha256(
                        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
                    ).hexdigest()

                corpus = {
                    "dataset_type": "synthetic",
                    "paragraphs": paragraphs,
                    "parsed_paragraphs_sha256": checksum(paragraphs),
                    "manifest": {
                        "paragraph_count": len(paragraphs),
                        "parsed_paragraphs_sha256": checksum(paragraphs),
                        "documents": [
                            {
                                "source": p["source"],
                                "paragraph_count": 1,
                                "parsed_paragraphs_sha256": checksum([p]),
                            }
                            for p in paragraphs
                        ],
                    },
                }
                corpus_path = Path(directory) / "corpus.json"
                corpus_path.write_text(json.dumps(corpus), "utf-8")
                checked = client.predict(
                    handle_file(corpus_path),
                    handle_file(result[3][2]),
                    api_name="/check_corpus_evidence",
                )
                assert "证据字段一致" in checked and "synthetic" in checked
                assert "手工构造" in checked and "不代表" in checked
                corpus["paragraphs"][0]["text"] = "tampered"
                corpus_path.write_text(json.dumps(corpus), "utf-8")
                failed_check = client.predict(
                    handle_file(corpus_path),
                    handle_file(result[3][2]),
                    api_name="/check_corpus_evidence",
                )
                assert "未完成核对" in failed_check and "校验值" in failed_check
                legacy = envelope["result"]
                for sentence in legacy["sentences"]:
                    for evidence in sentence["evidences"]:
                        evidence.pop("source")
                legacy_path = Path(directory) / "legacy.json"
                legacy_path.write_text(json.dumps(legacy), "utf-8")
                legacy_import = client.predict(handle_file(legacy_path), api_name="/import_result")
                assert "文献来源未知" in legacy_import[2]
                cleared = client.predict(api_name="/clear_result")
                assert cleared[1:] == ("", "", None)
                failed = client.predict("", None, 3, 0.7, api_name="/generate_review")
                assert "请输入问题" in failed[0] and failed[1:] == ("", "", None)
                bad = Path(directory) / "bad.json"
                bad.write_text("{", "utf-8")
                failed = client.predict(handle_file(bad), api_name="/import_result")
                assert "不是有效" in failed[0] and failed[1:] == ("", "", None)
                pdfs = []
                for i in range(5):
                    pdf = Path(directory) / f"synthetic-{i}.pdf"
                    pdf.write_bytes(b"%PDF-1.7\nsynthetic upload validation only")
                    pdfs.append(handle_file(pdf))
                duplicate = client.predict(
                    "重复来源检查", [pdfs[0], pdfs[0]], 3, 0.7, api_name="/generate_review"
                )
                assert "重复" in duplicate[0] and duplicate[1:] == ("", "", None)
                # Valid list reaches the selected backend, which must honestly fail
                # without models. These minimal headers are not real PDF/model data.
                uploaded = client.predict("多篇上传检查", pdfs, 3, 0.7, api_name="/generate_review")
                assert "生成未完成" in uploaded[0] and uploaded[1:] == ("", "", None)
                repeated = client.predict(api_name="/show_demo")
                assert repeated[1] == result[1]
                old_ids = set(re.findall(r'id="([^"]+)"', result[2]))
                new_ids = set(re.findall(r'id="([^"]+)"', repeated[2]))
                assert old_ids and new_ids and old_ids.isdisjoint(new_ids)
                assert Path(repeated[3][2]) != Path(result[3][2])
                print(
                    "PASS: 本地 HTTP 服务、后端环境自检、四种下载、JSON 回导、"
                    "多来源/旧格式回导、冻结语料核对/篡改拒绝、五文件上传、同名拦截、清空、失败状态、重复操作。"
                )
                print("仅使用手工演示数据；未验证浏览器视觉效果、真实模型或正式评测。")
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
