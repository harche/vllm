# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Device fault classification for the CUDA platform.

The classification answers one question for the orchestrator: can this
process keep using its GPU, or does it need a restart? It combines the CUDA
error class with the NVML recovery action for the worker's device, the same
signal the NVIDIA DRA driver uses for device health.

`CudaPlatformBase.classify_device_fault` builds the result. Another
accelerator adds its own `DeviceFault` subclass and implements the same
`Platform` hook.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch

from vllm.v1.fault_tolerance.utils import DeviceFault, exception_chain

# CUDA errors after which "the process must be terminated and relaunched",
# according to the cudaError_t documentation:
# https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__TYPES.html#_CPPv49cudaError
STICKY_CUDA_ERRORS = frozenset(
    {226, 700, 702, 710, 714, 715, 716, 717, 718, 719, 721, 810, 911}
)

# nvmlDeviceGpuRecoveryAction_t, as defined by NVML API version 13. The table
# is kept here because the pynvml copy in `vllm/third_party` (nvidia-ml-py
# 12.570.86) defines only values 0 to 4.
_NVML_GPU_RECOVERY_ACTIONS = {
    0: "NONE",
    1: "GPU_RESET",
    2: "NODE_REBOOT",
    3: "DRAIN_P2P",
    4: "DRAIN_AND_RESET",
    5: "RECOVER_IMEX_DOMAIN",
    6: "BUS_RESET",
    7: "SYSTEM_REBOOT",
}

# NVML GPU recovery actions that take the device away from the process. NONE
# and RECOVER_IMEX_DOMAIN leave the device usable. An action NVML reports that
# is newer than this code arrives as a number and does not require a restart,
# because marking a rank dead needs a value we can interpret.
RESTART_RECOVERY_ACTIONS = frozenset(
    {
        "GPU_RESET",
        "NODE_REBOOT",
        "DRAIN_P2P",
        "DRAIN_AND_RESET",
        "BUS_RESET",
        "SYSTEM_REBOOT",
    }
)


# NVML errors that mean the process cannot keep using the device, quoted from
# nvml.h:
# - NVML_ERROR_GPU_IS_LOST: "The GPU has fallen off the bus or has otherwise
#   become inaccessible"
# - NVML_ERROR_RESET_REQUIRED: "The GPU requires a reset before it can be used
#   again". nvml.h lists it only for calls that change device settings, so the
#   read-only query made here is not expected to return it. It is handled
#   because its meaning leaves no doubt.
RESTART_DRIVER_ERRORS = frozenset({"GPU_IS_LOST", "RESET_REQUIRED"})


class CudaDeviceState(NamedTuple):
    """What the driver reports about a worker's device after a fault."""

    # NVML GPU recovery action, or None when it cannot be queried.
    recovery_action: str | None = None
    # One of RESTART_DRIVER_ERRORS when NVML refused the query with it.
    driver_error: str | None = None


@dataclass(frozen=True)
class CudaFault(DeviceFault):
    """A CUDA fault: `error_code` is a `cudaError_t` value and
    `recovery_action` is the NVML GPU recovery action."""

    backend: ClassVar[str] = "cuda"
    status_fields: ClassVar[tuple[str, ...]] = (
        "error_code",
        "sticky",
        "recovery_action",
        "driver_error",
    )

    # Set when the driver refused the query with one of RESTART_DRIVER_ERRORS.
    # Such a device reports no recovery action, because it answers no queries
    # at all, so without this field a non-sticky error on it would look
    # retryable.
    driver_error: str | None = None

    @classmethod
    def from_cuda_error(cls, error_code: int, device_state: CudaDeviceState):
        return cls(
            error_code=error_code,
            sticky=error_code in STICKY_CUDA_ERRORS,
            recovery_action=device_state.recovery_action,
            driver_error=device_state.driver_error,
        )

    @property
    def restart_required(self) -> bool:
        """The process cannot keep using its device when any of these holds:
        the CUDA error is sticky, the driver reports the device lost or in
        need of a reset, or the driver recommends an action that takes the
        device away."""
        return (
            self.sticky
            or self.driver_error in RESTART_DRIVER_ERRORS
            or self.recovery_action in RESTART_RECOVERY_ACTIONS
        )

    def __str__(self) -> str:
        return f"{super().__str__()} driver_error={self.driver_error}"


def cuda_error_code(exc: BaseException) -> int | None:
    """The CUDA error code behind `exc`, read from `torch.AcceleratorError` in
    its cause chain. None if `exc` is not a CUDA error."""
    for cur in exception_chain(exc):
        if isinstance(cur, torch.AcceleratorError):
            return getattr(cur, "error_code", None)
    return None


def nvml_recovery_action_name(value: int) -> str:
    """Name an NVML GPU recovery action. A value newer than the table is
    reported as its number."""
    return _NVML_GPU_RECOVERY_ACTIONS.get(value, str(value))


def classify_cuda_fault(
    exc: BaseException | None,
    get_device_state: Callable[[], CudaDeviceState],
    probe_context: Callable[[], None],
) -> CudaFault | None:
    """Classify `exc` as a CUDA fault; None if it is not one.

    The error code comes from `exc`. When `exc` carries none, or when there
    is no exception, it comes from `probe_context`, a call into the CUDA
    runtime: a sticky error is returned by every later CUDA call in the
    process, so the probe finds one that first surfaced without a code.
    cuBLAS, cuDNN and NCCL report failures that way.

    `get_device_state` asks the driver about the worker's device. It is
    called only when there is an error code.
    """
    error_code = cuda_error_code(exc) if exc is not None else None
    if error_code is None:
        try:
            probe_context()
        except Exception as probe_exc:
            error_code = cuda_error_code(probe_exc)
    if error_code is None:
        return None
    return CudaFault.from_cuda_error(error_code, get_device_state())
