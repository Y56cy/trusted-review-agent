"""Exercise the real local Gradio HTTP API without models or browser automation.

Run with: python tests/integration/ui_smoke.py
This explicitly synthetic smoke test is not a real model or evaluation run.
"""

from __future__ import annotations

import json
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
                for filename in result[3]:
                    assert Path(filename).is_file() and Path(filename).stat().st_size > 0
                envelope = json.loads(Path(result[3][2]).read_text("utf-8"))
                assert envelope["metadata"]["is_demo"] is True
                imported = client.predict(handle_file(result[3][2]), api_name="/import_result")
                assert "来源未核验" in imported[0] and "手工构造" in imported[0]
                assert imported[1] == result[1]
                cleared = client.predict(api_name="/clear_result")
                assert cleared[1:] == ("", "", None)
                failed = client.predict("", None, 3, 0.7, api_name="/generate_review")
                assert "请输入问题" in failed[0] and failed[1:] == ("", "", None)
                bad = Path(directory) / "bad.json"
                bad.write_text("{", "utf-8")
                failed = client.predict(handle_file(bad), api_name="/import_result")
                assert "不是有效" in failed[0] and failed[1:] == ("", "", None)
                repeated = client.predict(api_name="/show_demo")
                assert repeated[1] == result[1]
                assert Path(repeated[3][2]) != Path(result[3][2])
                print("PASS: 本地 HTTP 服务、四种下载、JSON 回导、清空、失败状态、重复操作。")
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
