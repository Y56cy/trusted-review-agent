"""Offline backend selection and prerequisite diagnostics; never import model modules."""

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

import app


@pytest.fixture
def ready_environment(tmp_path, monkeypatch):
    source = tmp_path / "src"
    source.mkdir()
    (source / "nli").mkdir()
    (source / "match.py").write_text('_NLI_PATH = "nli"', encoding="utf-8")
    (tmp_path / "models" / "bge-small-zh-v1.5").mkdir(parents=True)
    versions = {"transformers": "4.46.3", "sentence-transformers": "3.2.1"}
    monkeypatch.setattr(app, "ROOT", tmp_path)
    monkeypatch.setattr(app, "SRC", source)
    monkeypatch.setattr(app.sys, "version_info", (3, 12, 14))
    monkeypatch.setattr(app.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(app.importlib.metadata, "version", versions.__getitem__)
    monkeypatch.setattr(app.socket, "create_connection", lambda *a, **kw: nullcontext())
    return source


def test_basic_check_can_pass_without_importing_models(ready_environment):
    assert app.environment_issues() == []
    report = app.environment_report()
    assert report["python_executable"] == sys.executable
    assert report["issues"] == []
    assert report["notice"] == app.CHECK_NOTICE
    assert "仍须使用真实 PDF" in app.format_environment_report(report)


@pytest.mark.parametrize("missing", ["numpy", "jieba", "rank_bm25"])
def test_preflight_covers_hybrid_retrieval_dependencies(ready_environment, monkeypatch, missing):
    monkeypatch.setattr(
        app.importlib.util, "find_spec", lambda name: None if name == missing else 1
    )
    issues = app.environment_issues()
    assert len(issues) == 1 and f"缺少 {missing}" in issues[0]


@pytest.mark.parametrize("distribution", ["transformers", "sentence-transformers"])
def test_preflight_rejects_changed_locked_versions(ready_environment, monkeypatch, distribution):
    original = app.importlib.metadata.version
    monkeypatch.setattr(
        app.importlib.metadata,
        "version",
        lambda name: "999.0" if name == distribution else original(name),
    )
    issues = app.environment_issues()
    assert len(issues) == 1 and distribution in issues[0] and "不要自动升级" in issues[0]


def test_missing_distribution_metadata_is_reported(ready_environment, monkeypatch):
    def unavailable(name):
        raise app.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(app.importlib.metadata, "version", unavailable)
    assert len(app.environment_issues()) == 2
    assert all("无法确认" in issue for issue in app.environment_issues())


def test_python_version_mismatch_is_reported(ready_environment, monkeypatch):
    monkeypatch.setattr(app.sys, "version_info", (3, 13, 0))
    assert app.environment_issues() == ["项目要求 Python 3.12；请使用兼容的核心 Python 环境。"]


@pytest.mark.parametrize(
    "source", ["", "_NLI_PATH = unavailable()", "not valid python!", "_NLI_PATH=3"]
)
def test_unavailable_nli_configuration_is_reported(ready_environment, source):
    (ready_environment / "match.py").write_text(source, encoding="utf-8")
    issues = app.environment_issues()
    assert len(issues) == 1 and "NLI" in issues[0]


def test_missing_model_directories_and_service_are_reported(ready_environment, monkeypatch):
    (ready_environment / "nli").rmdir()
    (app.ROOT / "models" / "bge-small-zh-v1.5").rmdir()

    def refused(*args, **kwargs):
        assert args[0] == ("127.0.0.1", 8080) and kwargs["timeout"] == 1
        raise ConnectionRefusedError()

    monkeypatch.setattr(app.socket, "create_connection", refused)
    issues = app.environment_issues()
    assert len(issues) == 3
    assert any("BGE" in issue for issue in issues)
    assert any("NLI" in issue for issue in issues)
    assert any("llama-server" in issue for issue in issues)


def sample_report(issues=None):
    return {
        "schema_version": 1,
        "python_executable": "/chosen/core/python",
        "python_version": "3.12.14",
        "issues": [] if issues is None else issues,
        "notice": app.CHECK_NOTICE,
    }


@pytest.mark.parametrize("backend_python", [None, "/chosen/core/python"])
@pytest.mark.parametrize("issues", [[], ["缺少权重"]])
def test_backend_probe_uses_selected_python_cwd_deadline_and_offline_flags(
    monkeypatch, backend_python, issues
):
    expected = sample_report(issues)

    def run(command, **kwargs):
        assert command == [
            backend_python or sys.executable,
            str(Path(app.__file__)),
            "--check",
            "--json",
        ]
        assert kwargs["cwd"] == app.SRC and kwargs["timeout"] == 7
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["env"]["PYTHONUTF8"] == "1"
        return SimpleNamespace(returncode=int(bool(issues)), stdout=json.dumps(expected), stderr="")

    monkeypatch.setattr(app.subprocess, "run", run)
    assert app.check_backend_environment(backend_python, timeout=7) == expected


@pytest.mark.parametrize("failure", [FileNotFoundError(), subprocess.TimeoutExpired("python", 15)])
def test_backend_probe_launch_failures_are_clear(monkeypatch, failure):
    def run(*args, **kwargs):
        raise failure

    monkeypatch.setattr(app.subprocess, "run", run)
    with pytest.raises(ValueError):
        app.check_backend_environment("/missing/python")
    assert app.check_environment("/missing/python").startswith("未完成：")


@pytest.mark.parametrize(
    "payload,returncode",
    [
        ("not json", 1),
        ("[]", 0),
        ("{}", 0),
        (json.dumps({**sample_report(), "schema_version": True}), 0),
        (json.dumps({**sample_report(), "issues": "wrong"}), 1),
        (json.dumps({**sample_report(), "issues": [True]}), 1),
        (json.dumps({**sample_report(), "python_executable": ""}), 0),
        (json.dumps({**sample_report(), "python_version": None}), 0),
        (json.dumps(sample_report()), 1),
        (json.dumps(sample_report(["problem"])), 0),
        (json.dumps(sample_report()), 2),
        ('{"issues":NaN}', 1),
    ],
)
def test_malformed_backend_diagnostics_fail_closed(monkeypatch, payload, returncode):
    monkeypatch.setattr(
        app.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=returncode, stdout=payload, stderr="broken"),
    )
    with pytest.raises(ValueError, match="环境自检未完成"):
        app.check_backend_environment()


@pytest.mark.parametrize("json_mode", [False, True])
def test_check_cli_honors_backend_python(monkeypatch, capsys, json_mode):
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.py", "--check", "--backend-python", "/chosen/core/python"]
        + (["--json"] if json_mode else []),
    )

    def probe(python):
        assert python == "/chosen/core/python"
        return sample_report(["缺少权重"])

    monkeypatch.setattr(app, "check_backend_environment", probe)
    monkeypatch.setattr(app, "environment_report", lambda: pytest.fail("Checked wrong environment"))
    assert app.main() == 1
    output = capsys.readouterr().out
    assert "/chosen/core/python" in output
    assert "缺少权重" in output
    if json_mode:
        assert json.loads(output) == sample_report(["缺少权重"])


@pytest.mark.parametrize("issues", [[], ["缺少权重"]])
def test_check_cli_current_interpreter_has_no_recursion(monkeypatch, capsys, issues):
    monkeypatch.setattr(sys, "argv", ["app.py", "--check", "--json"])
    monkeypatch.setattr(app, "environment_report", lambda: sample_report(issues))
    monkeypatch.setattr(app, "check_backend_environment", lambda *a: pytest.fail("Recursed"))
    assert app.main() == int(bool(issues))
    assert json.loads(capsys.readouterr().out) == sample_report(issues)


@pytest.mark.parametrize("json_mode", [False, True])
def test_check_cli_reports_launch_error_separately(monkeypatch, capsys, json_mode):
    monkeypatch.setattr(
        sys,
        "argv",
        ["app.py", "--check", "--backend-python", "missing"] + (["--json"] if json_mode else []),
    )

    def fail(*args):
        raise ValueError("无法启动核心 Python")

    monkeypatch.setattr(app, "check_backend_environment", fail)
    assert app.main() == 2
    output = capsys.readouterr().out
    assert "无法启动核心 Python" in output
    if json_mode:
        assert "error" in json.loads(output)


def test_json_cli_requires_check(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["app.py", "--json"])
    with pytest.raises(SystemExit) as exc:
        app.main()
    assert exc.value.code == 2


def test_real_selected_interpreter_probe_does_not_load_model_dependencies():
    # This machine lacks model assets. Check report transport, not model readiness.
    report = app.check_backend_environment(sys.executable)
    assert Path(report["python_executable"]).samefile(sys.executable)
    assert isinstance(report["issues"], list)
    assert report["notice"] == app.CHECK_NOTICE
