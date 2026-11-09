# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Classify a worker exception as a GPU fault.

The classification answers one question for the orchestrator: can this
process keep using its GPU, or does it need a restart? It combines the CUDA
error class with the NVML recovery action for the worker's device, the same
signal the NVIDIA DRA driver uses for device health.
"""

import json
import os
from dataclasses import asdict, dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# CUDA errors after which "any further CUDA work will return the same error"
# and "the process must be terminated and relaunched" (cudaError_t docs).
STICKY_CUDA_ERRORS = frozenset({700, 702, 710, 714, 715, 716, 717, 718, 719})

_RECOVERY_ACTIONS = {
    0: "NONE",
    1: "GPU_RESET",
    2: "NODE_REBOOT",
    3: "DRAIN_P2P",
    4: "DRAIN_AND_RESET",
}

TERMINATION_LOG = "/dev/termination-log"


@dataclass(frozen=True)
class GpuFault:
    cuda_error: int
    sticky: bool
    gpu_uuid: str | None = None
    recovery_action: str | None = None

    @property
    def restart_required(self) -> bool:
        return self.sticky or self.recovery_action not in (None, "NONE")

    def to_dict(self) -> dict:
        return asdict(self) | {"restart_required": self.restart_required}

    def __str__(self) -> str:
        return (
            f"GPU fault: cuda_error={self.cuda_error} sticky={self.sticky} "
            f"gpu={self.gpu_uuid} recovery_action={self.recovery_action}"
        )


def _find_accelerator_error(exc: BaseException) -> "torch.AcceleratorError | None":
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if isinstance(cur, torch.AcceleratorError):
            return cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return None


def _query_nvml(device_id: int) -> tuple[str | None, str | None]:
    """Return (uuid, recovery action) for a vLLM-local device id."""
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import import_pynvml

    if not current_platform.is_cuda():
        return None, None
    pynvml = import_pynvml()
    try:
        pynvml.nvmlInit()
    except Exception:
        return None, None
    try:
        index = current_platform.device_id_to_physical_device_id(device_id)
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        field = pynvml.nvmlDeviceGetFieldValues(
            handle, [pynvml.NVML_FI_DEV_GET_GPU_RECOVERY_ACTION]
        )[0]
        action = None
        if field.nvmlReturn == pynvml.NVML_SUCCESS:
            action = _RECOVERY_ACTIONS.get(field.value.uiVal, str(field.value.uiVal))
        return uuid, action
    except Exception:
        logger.debug("NVML query for GPU fault classification failed", exc_info=True)
        return None, None
    finally:
        pynvml.nvmlShutdown()


def classify_gpu_fault(exc: BaseException, device_id: int) -> GpuFault | None:
    """Classify ``exc`` raised on ``device_id``; None if it is not a GPU fault."""
    err = _find_accelerator_error(exc)
    code = getattr(err, "error_code", None)
    if code is None:
        return None
    uuid, action = _query_nvml(device_id)
    return GpuFault(
        cuda_error=code,
        sticky=code in STICKY_CUDA_ERRORS,
        gpu_uuid=uuid,
        recovery_action=action,
    )


def write_termination_log(faults: list[GpuFault]) -> None:
    """Record the faults in the Kubernetes termination message, if present."""
    if not faults or not os.path.exists(TERMINATION_LOG):
        return
    try:
        with open(TERMINATION_LOG, "w") as f:
            json.dump({"gpu_faults": [fault.to_dict() for fault in faults]}, f)
    except OSError:
        logger.debug("Could not write %s", TERMINATION_LOG, exc_info=True)
