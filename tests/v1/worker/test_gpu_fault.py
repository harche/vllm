# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU fault classification: is the worker's CUDA context still usable?"""

import json

import pytest
import torch

from vllm.v1.worker import gpu_fault
from vllm.v1.worker.gpu_fault import GpuFault, classify_gpu_fault


def _cuda_error(code: int) -> torch.AcceleratorError:
    err = torch.AcceleratorError(f"CUDA error {code}")
    err.error_code = code
    return err


@pytest.fixture
def nvml(monkeypatch):
    monkeypatch.setattr(gpu_fault, "_query_nvml", lambda _: ("GPU-0", "NONE"))


@pytest.mark.parametrize(
    "code, sticky",
    [(700, True), (710, True), (719, True), (2, False)],  # 2: out of memory
)
def test_classify_cuda_error(nvml, code, sticky):
    try:
        raise RuntimeError("step failed") from _cuda_error(code)
    except RuntimeError as exc:
        fault = classify_gpu_fault(exc, device_id=0)

    assert fault == GpuFault(code, sticky, "GPU-0", "NONE")
    assert fault.restart_required is sticky


def test_non_cuda_error_is_not_a_gpu_fault(nvml):
    assert classify_gpu_fault(ValueError("bad input"), device_id=0) is None
    assert classify_gpu_fault(torch.AcceleratorError("no code"), device_id=0) is None


def test_recovery_action_requires_restart():
    assert GpuFault(999, sticky=False, recovery_action="GPU_RESET").restart_required
    assert not GpuFault(999, sticky=False, recovery_action="NONE").restart_required


def test_write_termination_log(tmp_path, monkeypatch):
    log = tmp_path / "termination-log"
    log.touch()
    monkeypatch.setattr(gpu_fault, "TERMINATION_LOG", str(log))

    gpu_fault.write_termination_log([GpuFault(700, True, "GPU-0", "NONE")])

    assert json.loads(log.read_text()) == {
        "gpu_faults": [
            {
                "cuda_error": 700,
                "sticky": True,
                "gpu_uuid": "GPU-0",
                "recovery_action": "NONE",
                "restart_required": True,
            }
        ]
    }
