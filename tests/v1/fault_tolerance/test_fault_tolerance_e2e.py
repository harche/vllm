# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for the elastic fault-tolerance framework.

Requires nixl_ep FT hardware; gated behind `has_nixl_ep()`.
"""

import contextlib
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import psutil
import pytest
import requests

from tests.utils import RemoteOpenAIServer, multi_gpu_test
from vllm.utils.import_utils import has_nixl_ep
from vllm.utils.network_utils import get_open_port

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B")

# Fault-detection timeout budget:
# - CPU: Gloo DP allreduce timeout (30s) detects the dead peer.
# - nixl_ep: kernel masks the dead rank after Buffer's default timeout_ms=30000 (30s).
# - Eager FP8 quantization can wait for the GPU timeout before the next CPU sync.
# - Deadline (75s): both 30s waits + margin.
CPU_DISTRIBUTED_TIMEOUT_S = 30
FAULT_DETECTION_DEADLINE_S = 75


NUM_REDUNDANT_EXPERTS = 64

# Recovery after scale_down (mask + expert redistribution + weight reload)
# is expected to finish well within this deadline; the busy-loop wrapper
# itself waits engine_recovery_timeout_sec (120s) before giving up.
SCALE_DOWN_DEADLINE_S = 20


# Patches ``gpu.dp_utils.sync_cudagraph_and_dp_padding`` to fault on ``rank``
# once armed (via /collective_rpc). Gated on VLLM_FT_TEST_INJECT_FAULT.
# ``kind=raise`` raises a Python exception. ``kind=cuda_assert`` triggers a
# device-side assert, a sticky CUDA error, on the worker's GPU.
# ``local_rank=N`` arms only that tensor-parallel worker of the engine.
# ``at=entry`` faults before the patched function runs, so the worker never
# reaches the per-step TP barrier inside it.
_FAULT_INJECT_SITECUSTOMIZE = """\
import builtins
import os
import sys

_SPEC = os.environ.get("VLLM_FT_TEST_INJECT_FAULT")
_MODULE = "vllm.v1.worker.gpu.dp_utils"
_ATTR = "sync_cudagraph_and_dp_padding"
_WORKER_MODULE = "vllm.v1.worker.gpu_worker"

if _SPEC:
    _f = dict(kv.split("=", 1) for kv in _SPEC.split(","))
    _RANK = int(_f["rank"])
    _KIND = _f.get("kind", "raise")
    _LOCAL_RANK = _f.get("local_rank")
    _AT_ENTRY = _f.get("at") == "entry"
    _ARMED = [False]

    def _fault():
        if _KIND == "cuda_assert":
            import torch

            # "cuda" is the current device of this process, which each
            # worker sets to its own GPU.
            x = torch.zeros(4, device="cuda")
            x[torch.tensor([100], device="cuda")]
            torch.accelerator.synchronize()
        raise RuntimeError("FT test fault injection (rank=%d)" % _RANK)

    def _arm():
        _ARMED[0] = True

    def _patch(m):
        import inspect
        _orig = getattr(m, _ATTR)
        _sig = inspect.signature(_orig)
        def _wrapped(*args, **kwargs):
            bound = _sig.bind(*args, **kwargs)
            bound.apply_defaults()
            fire = bound.arguments.get("dp_rank") == _RANK and _ARMED[0]
            if fire:
                # One-shot: disarm before raising so post-retry steps pass.
                _ARMED[0] = False
                if _AT_ENTRY:
                    _fault()
            result = _orig(*args, **kwargs)
            if fire:
                _fault()
            return result

        setattr(m, _ATTR, _wrapped)

    _real_import = builtins.__import__

    def _hook(name, *a, **k):
        module = _real_import(name, *a, **k)
        m = sys.modules.get(_MODULE)
        # During vLLM's circular import the module lands in sys.modules before
        # its functions are defined; hasattr guards against patching too early.
        if (
            m is not None
            and hasattr(m, _ATTR)
            and not getattr(m, "_ft_patched", False)
        ):
            m._ft_patched = True
            _patch(m)
        # Expose arming to /collective_rpc via a Worker method.
        w = sys.modules.get(_WORKER_MODULE)
        if (
            w is not None
            and hasattr(w, "Worker")
            and not getattr(w, "_ft_arm_patched", False)
        ):
            w._ft_arm_patched = True
            w.Worker.arm_ft_fault_injection = lambda self, *a, **k: (
                _arm()
                if _LOCAL_RANK is None or self.local_rank == int(_LOCAL_RANK)
                else None
            )
        return module

    builtins.__import__ = _hook
"""


def _install_fault_injection(
    monkeypatch,
    tmp_path,
    rank: int,
    kind: str = "raise",
    local_rank: int | None = None,
    at_entry: bool = False,
) -> None:
    """Install a sitecustomize that faults on `rank` once armed.

    `local_rank` limits the fault to one tensor-parallel worker of the rank.
    `at_entry` faults before the DP sync instead of after it.

    Disarmed until the test calls /collective_rpc (needs VLLM_SERVER_DEV_MODE,
    set here).
    """
    site_dir = tmp_path / "ft_inject"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(_FAULT_INJECT_SITECUSTOMIZE)
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH",
        str(site_dir) + (os.pathsep + existing if existing else ""),
    )
    spec = f"rank={rank},kind={kind}"
    if local_rank is not None:
        spec += f",local_rank={local_rank}"
    if at_entry:
        spec += ",at=entry"
    monkeypatch.setenv("VLLM_FT_TEST_INJECT_FAULT", spec)
    monkeypatch.setenv("VLLM_SERVER_DEV_MODE", "1")


def _ft_server_args(extra_args: list[str] | None = None) -> list[str]:
    return [
        "--enforce-eager",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "2048",
        "--max-num-seqs",
        "128",
        "--enable-expert-parallel",
        "--all2all-backend",
        "nixl_ep",
        "--enable-fault-tolerance",
        "--cpu-distributed-timeout-seconds",
        str(CPU_DISTRIBUTED_TIMEOUT_S),
        "--fault-tolerance-config",
        '{"engine_recovery_timeout_sec": 120}',
        *(extra_args or []),
    ]


def _ft_manager(
    extra_args: list[str] | None = None, dp_size: int = 2, tp_size: int = 1
):
    """Build the shared DP+EP fault-tolerant server topology (one engine/server)."""
    from tests.v1.distributed.test_external_lb_dp import ExternalLBServerManager

    return ExternalLBServerManager(
        MODEL_NAME,
        dp_size,
        api_server_count=1,  # FT requires a single API server per engine
        base_server_args=_ft_server_args(extra_args),
        tp_size=tp_size,
    )


def _server_for_rank(servers, rank: int):
    """Locate the server for a DP rank."""
    for server, sargs in servers:
        if "--data-parallel-rank" in sargs:
            idx = sargs.index("--data-parallel-rank")
            if int(sargs[idx + 1]) == rank:
                return server
    raise AssertionError(f"no server found for DP rank {rank}")


def _complete(client):
    """Issue the one standard completion the tests use everywhere."""
    return client.completions.create(
        model=MODEL_NAME,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
        timeout=10.0,
    )


# Invalid responses count as wrong answers. Env overrides exist for tuning.
NUM_GSM8K_QUESTIONS = int(os.getenv("NUM_GSM8K_QUESTIONS", "256"))
EXPECTED_ACCURACY = float(os.getenv("EXPECTED_ACCURACY", "0.8"))
ACCURACY_TOL = float(os.getenv("ACCURACY_TOL", "0.08"))


def _run_gsm8k_eval(server, stage: str) -> float:
    """Run the gsm8k probe; assert the absolute bar, return the accuracy."""
    from tests.evals.gsm8k.gsm8k_eval import evaluate_gsm8k

    host_port = server.url_for("v1").split("//")[1].split("/")[0]
    host, port_str = host_port.rsplit(":", 1)
    results = evaluate_gsm8k(
        num_questions=NUM_GSM8K_QUESTIONS,
        num_shots=5,
        host=f"http://{host}",
        port=int(port_str),
        temperature=0.0,
        seed=42,
    )
    accuracy = results["accuracy"]
    print(
        f"[{stage}] GSM8K accuracy: {accuracy:.3f} "
        f"invalid_rate={results['invalid_rate']:.3f} "
        f"({results['num_questions']} questions)"
    )
    assert accuracy >= EXPECTED_ACCURACY, (
        f"[{stage}] gsm8k accuracy {accuracy:.3f} < {EXPECTED_ACCURACY}"
    )
    return accuracy


def _in_parallel(fn, servers) -> list:
    """Run `fn(server)` for all servers concurrently; return results in order."""
    with ThreadPoolExecutor(max_workers=len(servers)) as ex:
        return list(ex.map(fn, servers))


def _get_ft_status(server) -> dict:
    resp = requests.get(server.url_for("v1/fault_tolerance/status"), timeout=10)
    resp.raise_for_status()
    return resp.json()


def _assert_serving_and_healthy(servers) -> None:
    """Wait until every engine is healthy, then serve one request per server."""
    healthy = _wait_for_engines(
        list(servers), match_key="status", match_values={"healthy"}
    )
    assert all(healthy), healthy
    _in_parallel(lambda s: _complete(s.get_client()), servers)


def _apply_ft(
    server,
    instruction: str,
    params: dict | None = None,
    request_id: str | None = None,
) -> dict:
    """POST an FT instruction; assert it is accepted (202) and return the body."""
    resp = requests.post(
        server.url_for("v1/fault_tolerance/apply"),
        json={
            "instruction": instruction,
            "params": params or {},
            "request_id": request_id or str(uuid.uuid4()),
        },
        timeout=10,
    )
    assert resp.status_code == 202, resp.text
    return resp.json()


def _arm_ft_injection(servers) -> None:
    """Arm the fault injection in every engine's workers via /collective_rpc."""
    for server in servers:
        resp = requests.post(
            server.url_for("collective_rpc"),
            json={"method": "arm_ft_fault_injection"},
            timeout=10,
        )
        assert resp.status_code == 200, resp.text


def _kill_worker_process(server) -> None:
    """SIGKILL only the worker proc, leaving EngineCore and API server alive."""
    workers = []
    for p in psutil.Process(server.proc.pid).children(recursive=True):
        # Skip descendants that already exited: cmdline() raises ZombieProcess
        # if unreaped, NoSuchProcess if reaped. A dead worker still fails below.
        with contextlib.suppress(psutil.ZombieProcess, psutil.NoSuchProcess):
            if "Worker" in " ".join(p.cmdline()):
                workers.append(p)
    assert len(workers) == 1, f"expected 1 worker proc, found: {workers}"
    workers[0].kill()


def _wait_for_engines(
    servers: list[RemoteOpenAIServer],
    match_key: str,
    match_values: set[str],
    deadline_s: int = FAULT_DETECTION_DEADLINE_S,
) -> list[dict[str, Any] | None]:
    """Poll `/v1/fault_tolerance/status` until each server's engine status matches.

    A server matches when its engine-status dict has `match_key` equal to
    one of `match_values`. Returns one engine-status dict per server. Servers still
    unmatched after `deadline_s` get None.
    """
    results: dict[int, dict[str, Any]] = {}
    pending = dict(enumerate(servers))
    start = time.time()
    while pending and time.time() - start < deadline_s:
        for i, server in list(pending.items()):
            with contextlib.suppress(Exception):
                for engine_status in _get_ft_status(server)["engines"]:
                    if engine_status.get(match_key) in match_values:
                        results[i] = engine_status
                        del pending[i]
                        break
        if pending:
            time.sleep(1.0)
    return [results.get(i) for i in range(len(servers))]


@contextlib.contextmanager
def _driving(*servers):
    """Pump completions at each server in the background for the block's duration.

    Keeps every engine stepping into its failed component so a fault surfaces.
    Errors are expected once faulted and are ignored.
    """
    stop = threading.Event()

    def _drive(server):
        client = server.get_client()
        while not stop.is_set():
            with contextlib.suppress(Exception):
                _complete(client)
            time.sleep(0.2)

    threads = [threading.Thread(target=_drive, args=(s,), daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        yield
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)


def _wait_for_ft_failure(server, deadline_s: int) -> str | None:
    """Wait until `/v1/fault_tolerance/status` records the FT failure outcome."""
    engine_status = _wait_for_engines(
        [server],
        match_key="ft_state",
        match_values={"failed"},
        deadline_s=deadline_s,
    )[0]
    return engine_status.get("ft_error") if engine_status else None


def _servers_by_rank(servers, dp_size: int) -> dict[int, RemoteOpenAIServer]:
    """Map each DP rank to its server, so tests can index by rank not position."""
    return {r: _server_for_rank(servers, r) for r in range(dp_size)}


def _drive_to_faulted(
    servers_by_rank: dict[int, RemoteOpenAIServer],
    match_values: set[str],
) -> dict[int, dict[str, Any]]:
    """Drive every engine until each reports a matching fault status.

    Returns a `{rank: engine_status}` map. Asserts no rank hung (all matched
    within the deadline) so callers can assume every value is present.
    """
    ranks = sorted(servers_by_rank)
    all_servers = [servers_by_rank[r] for r in ranks]
    with _driving(*all_servers):
        statuses = _wait_for_engines(
            all_servers, match_key="status", match_values=match_values
        )
    faulted: dict[int, dict[str, Any]] = {}
    for rank, engine_status in zip(ranks, statuses):
        assert engine_status is not None, (
            f"rank {rank} did not report fault within "
            f"{FAULT_DETECTION_DEADLINE_S}s -- it likely hung"
        )
        faulted[rank] = engine_status
    return faulted


@pytest.mark.skipif(not has_nixl_ep(), reason="Requires nixl_ep all2all backend")
@multi_gpu_test(num_gpus=2)
def test_injected_fault_retry_recovers_all_ranks(monkeypatch, tmp_path):
    """An exception injected into the inference path drives full retry recovery.

    The fault is patched into `sync_cudagraph_and_dp_padding` on rank 1 via
    a generated `sitecustomize`; it raises on the engine's first DP-sync
    step after arming, so rank 1 goes UNHEALTHY and rank 0 times out on the
    DP allreduce. Both UNHEALTHY is the precondition for `retry`.

    The injection is armed (via /collective_rpc) only after the baseline
    accuracy eval; accuracy is compared before and after recovery, mirroring
    test_elastic_ep.py.
    """
    _install_fault_injection(monkeypatch, tmp_path, rank=1)
    dp_size = 2
    faulted_rank = 1
    with _ft_manager(dp_size=dp_size) as servers:
        assert len(servers) == dp_size
        servers_by_rank = _servers_by_rank(servers, dp_size)

        # 1. All engines healthy and serving.
        _assert_serving_and_healthy(list(servers_by_rank.values()))

        # 2. Baseline accuracy (the injection is still disarmed).
        initial_accuracy = _run_gsm8k_eval(servers_by_rank[0], "Initial")

        # 3. Arm, then drive: the injected rank faults on its next engine step.
        _arm_ft_injection(list(servers_by_rank.values()))
        faulted = _drive_to_faulted(servers_by_rank, match_values={"unhealthy"})

        # The rank that raised carries the fault info from its own exception.
        # A Python exception is not a device fault, so the rank stays retryable.
        assert faulted[faulted_rank].get("fault_info"), faulted[faulted_rank]
        assert "device_fault" not in faulted[faulted_rank], faulted[faulted_rank]

        # 4. retry every engine.
        round_id = str(uuid.uuid4())
        for server in servers_by_rank.values():
            _apply_ft(server, "retry", request_id=round_id)

        # 5. Recovery completes: every engine returns to healthy and serves again.
        _assert_serving_and_healthy(list(servers_by_rank.values()))

        # 6. Post-recovery accuracy vs the pre-fault baseline.
        recovered_accuracy = _run_gsm8k_eval(servers_by_rank[0], "After retry")
        assert recovered_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Accuracy dropped after retry: {recovered_accuracy:.3f} < "
            f"{initial_accuracy:.3f} - {ACCURACY_TOL}"
        )
        print(
            "\nAccuracy Summary:\n"
            f"  Initial:     {initial_accuracy:.3f}\n"
            f"  After retry: {recovered_accuracy:.3f} "
            f"(diff: {recovered_accuracy - initial_accuracy:+.3f})\n"
            f"  Tolerance:   {ACCURACY_TOL:.3f}"
        )


@pytest.mark.skipif(not has_nixl_ep(), reason="Requires nixl_ep all2all backend")
@multi_gpu_test(num_gpus=4)
@pytest.mark.parametrize("quantization", [None, "fp8_per_block"])
def test_scale_down_removes_dead_rank_and_recovers(quantization):
    """Preserve physical expert IDs and reload recovered placement after scale-down.

    Survivors keep serving with a smaller DP group, including processed FP8
    weights whose checkpoint layout differs from their runtime layout.

    SIGKILLing rank 0's worker leaves its EngineCore DEAD while the other ranks
    detect the peer fault and go UNHEALTHY. The orchestrator then sends
    `scale_down` with `removed_dp_ranks=[0]` to every survivor.


    Also verifies that a DEAD engine rejects `retry`: recovery is gated on
    UNHEALTHY, so trying `retry` on the victim records a rejection reason.

    Accuracy is measured before the fault and after scale_down; the survivor
    must stay within ACCURACY_TOL of the pre-fault baseline.
    """
    eplb_args = [
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        str(NUM_REDUNDANT_EXPERTS),
    ]
    if quantization is not None:
        eplb_args.extend(["--quantization", quantization])
    dp_size = 4
    victim_rank = 0
    with _ft_manager(eplb_args, dp_size=dp_size) as servers:
        assert len(servers) == dp_size
        servers_by_rank = _servers_by_rank(servers, dp_size)
        victim = servers_by_rank[victim_rank]
        survivor_ranks = [r for r in servers_by_rank if r != victim_rank]
        survivors = [servers_by_rank[r] for r in survivor_ranks]

        # 1. All engines healthy and serving.
        _assert_serving_and_healthy(list(servers_by_rank.values()))

        # 2. Baseline accuracy while the full DP group is healthy.
        initial_accuracy = _run_gsm8k_eval(servers_by_rank[0], "Initial")

        # 3. Kill the victim's worker; drive all engines into the fault.
        _kill_worker_process(victim)
        faulted = _drive_to_faulted(servers_by_rank, match_values={"dead", "unhealthy"})

        # 4. DEAD engine rejects retry: recovery requires UNHEALTHY.
        assert faulted[victim_rank]["status"] == "dead", faulted[victim_rank]
        _apply_ft(victim, "retry")
        ft_error = _wait_for_ft_failure(victim, FAULT_DETECTION_DEADLINE_S)
        assert ft_error is not None, (
            "rejection was never recorded in /v1/fault_tolerance/status"
        )
        assert "status is DEAD" in ft_error, ft_error

        # 5. scale_down sent to every survivor: remove the dead rank.
        round_id = str(uuid.uuid4())
        store_port = get_open_port()
        for server in survivors:
            _apply_ft(
                server,
                "scale_down",
                {
                    "removed_dp_ranks": [victim_rank],
                    "dp_store_port": store_port,
                    "dp_master_ip": "127.0.0.1",
                },
                request_id=round_id,
            )

        # 6. Recovery completes: all survivors are healthy and serving.
        recovered = _wait_for_engines(
            survivors,
            match_key="status",
            match_values={"healthy"},
            deadline_s=SCALE_DOWN_DEADLINE_S,
        )
        for rank, engine_status in zip(survivor_ranks, recovered):
            assert engine_status is not None, (
                f"survivor {rank} did not recover within {SCALE_DOWN_DEADLINE_S}s "
                "-- expert reload or DP-group reinit likely failed"
            )
        # Verify all survivors actually serve.
        for server in survivors:
            completion = _complete(server.get_client())
            assert completion.choices[0].text

        # 7. Post-recovery accuracy vs the pre-fault baseline.
        recovered_accuracy = _run_gsm8k_eval(survivors[0], "After scale down")
        assert recovered_accuracy >= initial_accuracy - ACCURACY_TOL, (
            f"Accuracy dropped after scale_down: {recovered_accuracy:.3f} < "
            f"{initial_accuracy:.3f} - {ACCURACY_TOL}"
        )
        print(
            "\nAccuracy Summary:\n"
            f"  Initial:          {initial_accuracy:.3f}\n"
            f"  After scale down: {recovered_accuracy:.3f} "
            f"(diff: {recovered_accuracy - initial_accuracy:+.3f})\n"
            f"  Tolerance:        {ACCURACY_TOL:.3f}"
        )


@pytest.mark.skipif(not has_nixl_ep(), reason="Requires nixl_ep all2all backend")
@multi_gpu_test(num_gpus=4)
def test_sticky_device_fault_reports_dead_and_scales_down(monkeypatch, tmp_path):
    """A sticky CUDA error makes the rank DEAD, so it is removed, not retried.

    A device-side assert is injected into rank 0's worker. CUDA documents the
    error as requiring the process to be relaunched, so `retry` cannot succeed
    on that rank. The engine must report `dead` with the worker's
    classification while its process is still running, and the survivors must
    recover with `scale_down`.
    """
    dp_size = 4
    victim_rank = 0
    _install_fault_injection(monkeypatch, tmp_path, victim_rank, kind="cuda_assert")
    eplb_args = [
        "--enable-eplb",
        "--eplb-config.num_redundant_experts",
        str(NUM_REDUNDANT_EXPERTS),
    ]
    with _ft_manager(eplb_args, dp_size=dp_size) as servers:
        assert len(servers) == dp_size
        servers_by_rank = _servers_by_rank(servers, dp_size)
        victim = servers_by_rank[victim_rank]
        survivor_ranks = [r for r in servers_by_rank if r != victim_rank]
        survivors = [servers_by_rank[r] for r in survivor_ranks]

        # 1. All engines healthy and serving.
        _assert_serving_and_healthy(list(servers_by_rank.values()))

        # 2. Arm, then drive: the victim's worker hits the device-side assert.
        # Only the victim is armed. It can fault before the next call returns,
        # and a survivor that is waiting on the dead rank answers nothing.
        _arm_ft_injection([victim])
        faulted = _drive_to_faulted(servers_by_rank, match_values={"dead", "unhealthy"})

        # 3. The victim reports DEAD with the worker's classification.
        status = faulted[victim_rank]
        assert status["status"] == "dead", status
        device_fault = dict(status["device_fault"])
        # The recovery action is null on a driver that does not report one.
        assert device_fault.pop("recovery_action") in ("NONE", None), status
        assert device_fault == {
            "backend": "cuda",
            "error_code": 710,
            "sticky": True,
            "driver_error": None,
            "restart_required": True,
        }, status
        assert status["fault_info"].endswith("; restart required"), status
        for rank in survivor_ranks:
            assert faulted[rank]["status"] == "unhealthy", faulted[rank]
            assert "device_fault" not in faulted[rank], faulted[rank]

        # 4. The DEAD engine rejects retry.
        _apply_ft(victim, "retry")
        ft_error = _wait_for_ft_failure(victim, FAULT_DETECTION_DEADLINE_S)
        assert ft_error is not None, (
            "rejection was never recorded in /v1/fault_tolerance/status"
        )
        assert "status is DEAD" in ft_error, ft_error

        # 5. scale_down sent to every survivor: remove the dead rank.
        round_id = str(uuid.uuid4())
        store_port = get_open_port()
        for server in survivors:
            _apply_ft(
                server,
                "scale_down",
                {
                    "removed_dp_ranks": [victim_rank],
                    "dp_store_port": store_port,
                    "dp_master_ip": "127.0.0.1",
                },
                request_id=round_id,
            )

        # 6. Recovery completes: all survivors are healthy and serving.
        recovered = _wait_for_engines(
            survivors,
            match_key="status",
            match_values={"healthy"},
            deadline_s=SCALE_DOWN_DEADLINE_S,
        )
        for rank, engine_status in zip(survivor_ranks, recovered):
            assert engine_status is not None, (
                f"survivor {rank} did not recover within {SCALE_DOWN_DEADLINE_S}s"
            )
        for server in survivors:
            completion = _complete(server.get_client())
            assert completion.choices[0].text


@pytest.mark.skipif(not has_nixl_ep(), reason="Requires nixl_ep all2all backend")
@multi_gpu_test(num_gpus=4)
def test_sticky_device_fault_on_tensor_parallel_sibling_reports_dead(
    monkeypatch, tmp_path
):
    """A sticky CUDA error on a TP worker that returns no output makes the rank DEAD.

    With TP=2 only local rank 0 returns the step's output. The device-side
    assert is injected into local rank 1 before the per-step TP barrier, so
    local rank 0 fails at the barrier with an error that is not a device
    fault. The engine has to ask both workers to report the classification.
    """
    dp_size = 2
    victim_rank = 0
    _install_fault_injection(
        monkeypatch,
        tmp_path,
        victim_rank,
        kind="cuda_assert",
        local_rank=1,
        at_entry=True,
    )
    with _ft_manager(dp_size=dp_size, tp_size=2) as servers:
        assert len(servers) == dp_size
        servers_by_rank = _servers_by_rank(servers, dp_size)
        victim = servers_by_rank[victim_rank]

        # 1. All engines healthy and serving.
        _assert_serving_and_healthy(list(servers_by_rank.values()))

        # 2. Arm the victim only, then drive until the barrier times out.
        _arm_ft_injection([victim])
        faulted = _drive_to_faulted(servers_by_rank, match_values={"dead", "unhealthy"})

        # 3. The victim reports DEAD with the classification of the worker
        # that returned no output.
        status = faulted[victim_rank]
        assert status["status"] == "dead", status
        device_fault = dict(status["device_fault"])
        # The recovery action is null on a driver that does not report one.
        assert device_fault.pop("recovery_action") in ("NONE", None), status
        assert device_fault == {
            "backend": "cuda",
            "error_code": 710,
            "sticky": True,
            "driver_error": None,
            "restart_required": True,
        }, status
        survivor = faulted[1 - victim_rank]
        assert survivor["status"] == "unhealthy", survivor
        assert "device_fault" not in survivor, survivor
