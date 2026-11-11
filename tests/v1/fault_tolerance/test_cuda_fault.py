# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA fault classification: is the worker's CUDA context still usable?"""

from dataclasses import dataclass
from typing import ClassVar

import pytest
import torch

from vllm.platforms.interface import Platform
from vllm.v1.fault_tolerance import cuda_fault
from vllm.v1.fault_tolerance import utils as ft_utils
from vllm.v1.fault_tolerance.cuda_fault import (
    CudaDeviceState,
    CudaFault,
    classify_cuda_fault,
    cuda_error_code,
    nvml_recovery_action_name,
)


def _cuda_error(code: int) -> torch.AcceleratorError:
    err = torch.AcceleratorError(f"CUDA error {code}")
    err.error_code = code
    return err


def _usable_context() -> None:
    """A context probe that finds the CUDA context usable."""


def _poisoned_context(code: int):
    """A context probe that finds the context left unusable by `code`."""

    def probe() -> None:
        raise _cuda_error(code)

    return probe


def test_cuda_error_code_is_read_through_the_cause_chain():
    try:
        raise RuntimeError("step failed") from _cuda_error(700)
    except RuntimeError as exc:
        assert cuda_error_code(exc) == 700


def test_non_cuda_error_has_no_error_code():
    assert cuda_error_code(ValueError("bad input")) is None
    assert cuda_error_code(torch.AcceleratorError("no code")) is None


@pytest.mark.parametrize(
    "code, sticky",
    [(700, True), (710, True), (719, True), (226, True), (2, False)],  # 2: OOM
)
def test_sticky_error_requires_restart(code, sticky):
    fault = CudaFault.from_cuda_error(code, CudaDeviceState("NONE"))

    assert fault == CudaFault(code, sticky, "NONE")
    assert fault.restart_required is sticky


def test_sticky_errors_match_cuda_documentation():
    # Codes documented as "the process must be terminated and relaunched".
    assert {
        226,
        700,
        702,
        710,
        714,
        715,
        716,
        717,
        718,
        719,
        721,
        810,
        911,
    } == cuda_fault.STICKY_CUDA_ERRORS


@pytest.mark.parametrize(
    "recovery_action, restart_required",
    [
        ("NONE", False),
        ("GPU_RESET", True),
        ("NODE_REBOOT", True),
        ("DRAIN_P2P", True),
        ("DRAIN_AND_RESET", True),
        ("RECOVER_IMEX_DOMAIN", False),
        ("BUS_RESET", True),
        ("SYSTEM_REBOOT", True),
        ("8", False),  # a value newer than the table
        (None, False),  # NVML unavailable
    ],
)
def test_recovery_action_decides_restart_for_a_non_sticky_error(
    recovery_action, restart_required
):
    fault = CudaFault.from_cuda_error(2, CudaDeviceState(recovery_action))

    assert fault.recovery_action == recovery_action
    assert fault.restart_required is restart_required


def test_sticky_error_requires_restart_whatever_the_recovery_action():
    assert CudaFault.from_cuda_error(700, CudaDeviceState("8")).restart_required


def test_recovery_action_table_matches_nvml():
    assert cuda_fault._NVML_GPU_RECOVERY_ACTIONS == {
        0: "NONE",
        1: "GPU_RESET",
        2: "NODE_REBOOT",
        3: "DRAIN_P2P",
        4: "DRAIN_AND_RESET",
        5: "RECOVER_IMEX_DOMAIN",
        6: "BUS_RESET",
        7: "SYSTEM_REBOOT",
    }


def test_unknown_recovery_action_is_reported_as_its_number():
    assert nvml_recovery_action_name(1) == "GPU_RESET"
    assert nvml_recovery_action_name(8) == "8"


def test_classify_cuda_fault_combines_error_code_and_recovery_action():
    try:
        raise RuntimeError("step failed") from _cuda_error(2)
    except RuntimeError as exc:
        fault = classify_cuda_fault(
            exc, lambda: CudaDeviceState("GPU_RESET"), _usable_context
        )

    assert fault == CudaFault(error_code=2, sticky=False, recovery_action="GPU_RESET")
    assert fault.to_dict()["backend"] == "cuda"
    assert fault.restart_required


def test_classify_cuda_fault_without_a_driver_answer():
    """When the recovery action cannot be queried, the CUDA error class
    alone still decides."""
    fault = classify_cuda_fault(_cuda_error(700), CudaDeviceState, _usable_context)

    assert fault == CudaFault(error_code=700, sticky=True, recovery_action=None)
    assert fault.restart_required


def test_driver_is_not_queried_for_an_error_that_is_not_a_device_fault():
    queried = []

    def get_device_state():
        queried.append(True)
        return CudaDeviceState("NONE")

    fault = classify_cuda_fault(
        ValueError("bad input"), get_device_state, _usable_context
    )

    assert fault is None
    assert not queried


def test_error_without_a_cuda_code_is_classified_from_the_context():
    """CuBLAS and NCCL failures reach Python without a CUDA error code. A
    sticky error behind one is still found, because the context returns it."""
    exc = RuntimeError("CUDA error: CUBLAS_STATUS_EXECUTION_FAILED")

    fault = classify_cuda_fault(exc, CudaDeviceState, _poisoned_context(710))

    assert fault == CudaFault(error_code=710, sticky=True, recovery_action=None)
    assert fault.restart_required


def test_context_is_not_probed_when_the_error_has_a_cuda_code():
    def probe() -> None:
        raise AssertionError("probed")

    fault = classify_cuda_fault(_cuda_error(2), CudaDeviceState, probe)

    assert fault == CudaFault(error_code=2, sticky=False, recovery_action=None)


def test_probe_failure_that_is_not_a_cuda_error_leaves_the_fault_unclassified():
    def probe() -> None:
        raise OSError("probe could not run")

    assert classify_cuda_fault(ValueError("x"), CudaDeviceState, probe) is None


@pytest.mark.parametrize("driver_error", ["GPU_IS_LOST", "RESET_REQUIRED"])
def test_driver_error_requires_restart_even_for_a_non_sticky_error(driver_error):
    """A device the driver cannot reach reports no recovery action, so the
    driver error has to carry the decision."""
    fault = classify_cuda_fault(
        _cuda_error(2),
        lambda: CudaDeviceState(driver_error=driver_error),
        _usable_context,
    )

    assert fault == CudaFault(
        error_code=2, sticky=False, recovery_action=None, driver_error=driver_error
    )
    assert fault.restart_required
    assert fault.to_dict()["driver_error"] == driver_error


def test_restart_driver_errors_are_the_two_nvml_names():
    assert {"GPU_IS_LOST", "RESET_REQUIRED"} == cuda_fault.RESTART_DRIVER_ERRORS


def test_backend_fault_without_a_restart_rule_fails_at_class_definition():
    """A backend that leaves the restart rule out is rejected at import, not
    at the first fault."""
    with pytest.raises(TypeError, match="restart_required"):

        class IncompleteFault(ft_utils.DeviceFault):
            backend = "incomplete"


def test_backend_fault_without_a_backend_name_fails_at_class_definition():
    with pytest.raises(TypeError, match="backend"):

        class UnnamedFault(ft_utils.DeviceFault):
            @property
            def restart_required(self) -> bool:
                return False


def test_backend_fault_without_published_fields_fails_at_class_definition():
    with pytest.raises(TypeError, match="status_fields"):

        class UnlistedFault(ft_utils.DeviceFault):
            backend = "unlisted"

            @property
            def restart_required(self) -> bool:
                return False


def test_backend_fault_publishing_an_unknown_field_fails_at_class_definition():
    with pytest.raises(TypeError, match="serial"):

        class MisnamedFault(ft_utils.DeviceFault):
            backend = "misnamed"
            status_fields = ("error_code", "serial")

            @property
            def restart_required(self) -> bool:
                return False


def test_field_a_backend_does_not_list_stays_out_of_the_status():
    """The status API publishes a field only when the backend names it, so
    a hardware identifier added to a fault is not exposed by default."""

    @dataclass(frozen=True)
    class IdentifiedFault(ft_utils.DeviceFault):
        backend: ClassVar[str] = "identified"
        status_fields: ClassVar[tuple[str, ...]] = ("error_code", "sticky")

        device_uuid: str = ""

        @property
        def restart_required(self) -> bool:
            return False

    fault = IdentifiedFault(error_code=1, sticky=False, device_uuid="GPU-1234")

    assert fault.to_dict() == {
        "backend": "identified",
        "error_code": 1,
        "sticky": False,
        "restart_required": False,
    }


def test_context_probe_uses_the_device_index_recorded_at_startup(monkeypatch):
    """With device UUIDs in CUDA_VISIBLE_DEVICES the index is found through
    NVML, which a lost device fails. The probe must not depend on it."""
    from vllm.platforms.cuda import CudaPlatformBase

    class LostDevicePlatform(CudaPlatformBase):
        @classmethod
        def logical_device_id_to_visible_device_id(cls, device_id: int) -> int:
            raise RuntimeError("NVML: GPU is lost")

        @classmethod
        def get_device_fault_state(cls, device_id: int = 0) -> CudaDeviceState:
            return CudaDeviceState(driver_error="GPU_IS_LOST")

    probed = []

    def synchronize(device=None) -> None:
        probed.append(device)
        raise _cuda_error(719)

    monkeypatch.setattr(torch.accelerator, "synchronize", synchronize)
    exc = RuntimeError("NCCL error: unhandled cuda error")

    fault = LostDevicePlatform.classify_device_fault(
        exc, device_id=0, visible_device_id=3
    )

    assert probed == [3]
    assert fault == CudaFault(719, sticky=True, driver_error="GPU_IS_LOST")


def test_platforms_do_not_classify_faults_by_default():
    """A CUDA error code means nothing on a platform that has not said how
    to read it, so other accelerators report no classification."""
    assert (
        Platform.classify_device_fault(
            _cuda_error(700), device_id=0, visible_device_id=0
        )
        is None
    )


STICKY = CudaFault(700, True, "NONE")
RESET = CudaFault(2, False, "GPU_RESET")
TRANSIENT = CudaFault(2, False, "NONE")


@pytest.mark.parametrize(
    "faults, reported",
    [
        ([], None),
        ([TRANSIENT], TRANSIENT),
        ([TRANSIENT, STICKY], STICKY),
        ([TRANSIENT, RESET, STICKY], RESET),
    ],
)
def test_engine_reports_the_fault_that_requires_a_restart(faults, reported):
    assert ft_utils.select_device_fault(faults) is reported


def test_fault_info_is_built_from_typed_fields_only():
    exc = RuntimeError("secret /models/path leaked by a worker")

    assert ft_utils.describe_fault(exc, None) == "RuntimeError"
    assert ft_utils.describe_fault(exc, TRANSIENT) == (
        "RuntimeError: device fault: backend=cuda error_code=2 sticky=False "
        "recovery_action=NONE driver_error=None"
    )
    assert ft_utils.describe_fault(exc, STICKY) == (
        "RuntimeError: device fault: backend=cuda error_code=700 sticky=True "
        "recovery_action=NONE driver_error=None; restart required"
    )


def test_status_payload_has_no_device_identifier():
    assert STICKY.to_dict() == {
        "backend": "cuda",
        "error_code": 700,
        "sticky": True,
        "recovery_action": "NONE",
        "driver_error": None,
        "restart_required": True,
    }
