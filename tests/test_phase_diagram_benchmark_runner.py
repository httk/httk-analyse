"""Focused checks for the guarded phase-diagram benchmark supervisor."""

import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCHMARKS = Path(__file__).parents[1] / "benchmarks"
RUNNER = BENCHMARKS / "run_phase_diagram_benchmarks.py"
_TIMEOUT = shutil.which("timeout")
_GUARD_AVAILABLE = (
    sys.platform == "linux"
    and _TIMEOUT is not None
    and "GNU coreutils" in subprocess.run([_TIMEOUT, "--version"], capture_output=True, text=True, check=False).stdout
)
pytestmark = pytest.mark.skipif(not _GUARD_AVAILABLE, reason="guarded measurement requires Linux and GNU timeout")


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch):
    """Load the standalone script with its sibling worker import available."""
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location("phase_diagram_benchmark_runner", RUNNER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def _arguments(tmp_path: Path, *, phase_lines: bool = False, resume: bool = False, repeats: int = 1) -> SimpleNamespace:
    return SimpleNamespace(
        json_path=tmp_path / "results.json",
        report=tmp_path / "results.md",
        resume=resume,
        report_only=False,
        phase_lines=phase_lines,
        repeats=repeats,
        atom_count=3,
        warmups=0,
        max_rss_gb=1.0,
        as_gb=1.0,
        timeout=1.0,
        threads=1,
        httk_solver="simplex",
    )


def _metadata(repeats: int) -> dict[str, object]:
    return {"parameters": {"repeats": repeats}, "sources": {"source": "fixed"}}


def _result() -> dict[str, object]:
    return {
        "status": "ok",
        "seconds": 0.01,
        "peak_rss_bytes": 1024,
        "summary": {
            "stable_indices": [0],
            "hull_energies": [0.0],
            "max_composition_residual": 0.0,
            "max_energy_residual": 0.0,
            "max_weight_sum_error": 0.0,
            "min_weight": 1.0,
        },
        "error": None,
        "diagnostic": "",
        "returncode": 0,
    }


def _write_worker(path: Path, source: str) -> Path:
    path.write_text(source, encoding="utf-8")
    return path


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _process_is_running(pid: int) -> bool:
    """Treat an unreaped zombie as exited while waiting for guard cleanup."""
    if not _process_exists(pid):
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").rsplit(")", 1)[1].lstrip()[0]
    except FileNotFoundError:
        return True  # A host-mounted /proc may not expose a namespaced child PID.
    return state != "Z"


def _allow_execute(monkeypatch: pytest.MonkeyPatch, runner) -> None:
    monkeypatch.setattr(runner.sys, "platform", "linux")
    monkeypatch.setattr(runner.shutil, "which", lambda name: "timeout")


def test_measurement_returns_worker_success_and_peak(tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch) -> None:
    worker = _write_worker(
        tmp_path / "success.py",
        """import json
import sys
from pathlib import Path
Path(sys.argv[2]).write_text(json.dumps({
    'status': 'ok', 'seconds': 0.125, 'peak_rss_bytes': 4321, 'summary': {}, 'error': None
}), encoding='utf-8')
""",
    )
    monkeypatch.setattr(runner, "WORKER", worker)

    result = runner._run_measurement({}, _arguments(tmp_path))

    assert result["status"] == "ok"
    assert result["seconds"] == 0.125
    assert result["peak_rss_bytes"] == 4321


def test_process_is_running_recognizes_sleeping_child() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        assert _process_is_running(child.pid)
    finally:
        child.terminate()
        child.wait(timeout=1)


def test_supervisor_cli_startup_does_not_import_scientific_packages() -> None:
    probe = f"""import runpy
import sys
sys.path.insert(0, {str(BENCHMARKS)!r})
sys.argv = [{str(RUNNER)!r}, '--help']
try:
    runpy.run_path({str(RUNNER)!r}, run_name='__main__')
except SystemExit as exc:
    assert exc.code == 0
print('LOADED:' + ','.join(sorted(name for name in sys.modules if name == 'numpy' or name == 'scipy' or name == 'ase' or name.startswith('httk.atomistic') or name.startswith('httk.analyse'))))
"""

    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[-1] == "LOADED:"


def test_parse_args_defaults_to_simplex_and_rejects_unknown_solver(runner) -> None:
    assert runner._parse_args([]).httk_solver == "simplex"
    with pytest.raises(SystemExit):
        runner._parse_args(["--httk-solver", "unknown"])


def test_execute_propagates_httk_solver_in_worker_request(
    tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    arguments.httk_solver = "highs"
    _allow_execute(monkeypatch, runner)
    case = {"sweep": "tiny", "species": 2, "phases": 3, "seed": 1}
    metadata = _metadata(arguments.repeats)
    requests: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "_cases", lambda args: [case])
    monkeypatch.setattr(runner, "_metadata", lambda args: metadata)
    monkeypatch.setattr(runner, "_source_state", lambda: metadata["sources"])
    monkeypatch.setattr(runner, "_run_measurement", lambda request, args: requests.append(request) or _result())

    assert runner._execute(arguments) == 0
    assert requests and all(request["solver"] == "highs" for request in requests)


def test_metadata_only_requires_highspy_for_highs(runner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.httk_solver = "highs"
    monkeypatch.setattr(runner, "_source_state", lambda: {"source": "fixed"})

    def missing_highspy(name: str) -> str:
        if name == "highspy":
            raise runner.importlib.metadata.PackageNotFoundError(name)
        return "installed"

    monkeypatch.setattr(runner.importlib.metadata, "version", missing_highspy)
    with pytest.raises(RuntimeError, match=r"highspy.*benchmark,highs"):
        runner._metadata(arguments)


def test_report_old_metadata_defaults_solver_to_simplex(runner) -> None:
    payload = {
        "started_at": "now",
        "metadata": {"parameters": {}, "sources": {}},
        "cases": [],
        "metrics": [],
        "results": [],
    }
    assert "HTTK solver: simplex." in runner._report(payload)


def test_measurement_timeout_kills_worker_descendants(tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch) -> None:
    pid_path = tmp_path / "child.pid"
    worker = _write_worker(
        tmp_path / "timeout.py",
        f"""import subprocess
import sys
import time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path({str(pid_path)!r}).write_text(str(child.pid), encoding='utf-8')
time.sleep(30)
""",
    )
    monkeypatch.setattr(runner, "WORKER", worker)
    arguments = _arguments(tmp_path)
    arguments.timeout = 1.0
    child_pid = None
    try:
        result = runner._run_measurement({}, arguments)
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 0.5
        while _process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)

        assert result["status"] == "timeout"
        assert not _process_is_running(child_pid)
    finally:
        if child_pid is None and pid_path.exists():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
        if child_pid is not None and _process_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_measurement_enforces_address_space_without_host_oom(
    tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker = _write_worker(
        tmp_path / "memory.py",
        """import json
import sys
from pathlib import Path
try:
    bytearray(512 * 2**20)
except MemoryError:
    result = {'status': 'memory_limit', 'seconds': None, 'peak_rss_bytes': 1, 'summary': {}, 'error': 'MemoryError'}
else:
    result = {'status': 'error', 'seconds': None, 'peak_rss_bytes': 1, 'summary': {}, 'error': 'limit not applied'}
Path(sys.argv[2]).write_text(json.dumps(result), encoding='utf-8')
""",
    )
    monkeypatch.setattr(runner, "WORKER", worker)
    arguments = _arguments(tmp_path)
    arguments.as_gb = 0.2

    result = runner._run_measurement({}, arguments)

    assert result["status"] == "memory_limit"
    assert result["error"] == "MemoryError"


def test_measurement_interrupt_terminates_worker_group(tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch) -> None:
    pid_path = tmp_path / "interrupt-child.pid"
    worker = _write_worker(
        tmp_path / "interrupt.py",
        f"""import os
import signal
import subprocess
import sys
import time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
Path({str(pid_path)!r}).write_text(str(child.pid), encoding='utf-8')
os.kill({os.getpid()}, signal.SIGINT)
time.sleep(30)
""",
    )
    monkeypatch.setattr(runner, "WORKER", worker)
    arguments = _arguments(tmp_path)
    arguments.timeout = 30.0
    child_pid = None
    try:
        with pytest.raises(KeyboardInterrupt):
            runner._run_measurement({}, arguments)
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 0.5
        while _process_is_running(child_pid) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not _process_is_running(child_pid)
    finally:
        if child_pid is None and pid_path.exists():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
        if child_pid is not None and _process_is_running(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_execute_checkpoints_each_sample_and_resumes_phase_lines(
    tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path, phase_lines=True, repeats=2)
    _allow_execute(monkeypatch, runner)
    case = {"sweep": "tiny", "species": 2, "phases": 3, "seed": 1}
    metadata = _metadata(arguments.repeats)
    monkeypatch.setattr(runner, "_cases", lambda args: [case])
    monkeypatch.setattr(runner, "_metadata", lambda args: metadata)
    monkeypatch.setattr(runner, "_source_state", lambda: metadata["sources"])
    calls: list[str] = []

    def interrupt_after_first(request, args):
        calls.append(request["metric"])
        if len(calls) == 2:
            raise KeyboardInterrupt
        return _result()

    monkeypatch.setattr(runner, "_run_measurement", interrupt_after_first)
    with pytest.raises(KeyboardInterrupt):
        runner._execute(arguments)

    checkpoint = json.loads(arguments.json_path.read_text(encoding="utf-8"))
    assert checkpoint["metrics"] == ["httk", "ase_construct", "ase_full", "httk_phase_lines"]
    assert [(record["metric"], record["repeat"]) for record in checkpoint["results"]] == [("httk", 0)]

    arguments.resume = True
    monkeypatch.setattr(runner, "_run_measurement", lambda request, args: _result())
    assert runner._execute(arguments) == 0
    resumed = json.loads(arguments.json_path.read_text(encoding="utf-8"))
    assert len(resumed["results"]) == 8
    assert sum(record["metric"] == "httk" for record in resumed["results"]) == 2


def test_execute_does_not_clobber_checkpoint_or_resume_changed_metadata(
    tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    _allow_execute(monkeypatch, runner)
    case = {"sweep": "tiny", "species": 2, "phases": 3, "seed": 1}
    metadata = _metadata(arguments.repeats)
    monkeypatch.setattr(runner, "_cases", lambda args: [case])
    monkeypatch.setattr(runner, "_metadata", lambda args: metadata)
    arguments.json_path.write_text('{"saved": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="exists"):
        runner._execute(arguments)
    assert arguments.json_path.read_text(encoding="utf-8") == '{"saved": true}\n'

    arguments.json_path.unlink()
    arguments.report.write_text("saved report\n", encoding="utf-8")
    with pytest.raises(ValueError, match="report path"):
        runner._execute(arguments)
    assert arguments.report.read_text(encoding="utf-8") == "saved report\n"
    assert not arguments.json_path.exists()

    arguments.resume = True
    checkpoint = {"format": 1, "metadata": {"changed": True}, "cases": [case], "results": []}
    arguments.json_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        runner._execute(arguments)
    assert json.loads(arguments.json_path.read_text(encoding="utf-8")) == checkpoint


def test_execute_invalidates_and_checkpoints_source_changed_after_worker(
    tmp_path: Path, runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _arguments(tmp_path)
    _allow_execute(monkeypatch, runner)
    case = {"sweep": "tiny", "species": 2, "phases": 3, "seed": 1}
    metadata = _metadata(arguments.repeats)
    source_states = iter((metadata["sources"], {"source": "changed"}))
    calls: list[str] = []
    monkeypatch.setattr(runner, "_cases", lambda args: [case])
    monkeypatch.setattr(runner, "_metadata", lambda args: metadata)
    monkeypatch.setattr(runner, "_source_state", lambda: next(source_states))
    monkeypatch.setattr(runner, "_run_measurement", lambda request, args: calls.append(request["metric"]) or _result())

    with pytest.raises(RuntimeError, match="source changed while worker ran"):
        runner._execute(arguments)

    checkpoint = json.loads(arguments.json_path.read_text(encoding="utf-8"))
    assert calls == ["httk"]
    assert checkpoint["results"][0]["status"] == "invalidated"
    assert checkpoint["results"][0]["seconds"] is None
    assert checkpoint["results"][0]["error"] == "source changed while worker ran"


def test_agreement_reports_membership_separately_from_residual_checks(runner) -> None:
    summary = {
        "max_composition_residual": 1e-9,
        "max_energy_residual": 2e-9,
        "max_weight_sum_error": 3e-9,
        "min_weight": 0.1,
        "hull_energies": [0.0, 1.0],
    }
    records = [
        {"status": "ok", "metric": "httk", "summary": dict(summary, stable_indices=[0])},
        {"status": "ok", "metric": "ase_construct", "summary": {"stable_indices": [1]}},
        {"status": "ok", "metric": "ase_full", "summary": dict(summary, stable_indices=[1])},
    ]

    checks = runner._agreement(records)

    assert checks["max_energy_residual"] == 2e-9
    assert checks["stable_membership_matches"] is False
    assert checks["numerical_checks_pass"] is True
