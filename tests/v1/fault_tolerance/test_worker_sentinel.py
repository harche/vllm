# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""WorkerSentinel: recording a device fault and handing it to the engine."""

from types import SimpleNamespace

from vllm.config import ParallelConfig
from vllm.v1.fault_tolerance.cuda_fault import CudaFault
from vllm.v1.fault_tolerance.utils import DeviceFault, FaultToleranceRequest
from vllm.v1.worker.sentinel.gpu_worker_sentinel import WorkerSentinel

STICKY = CudaFault(error_code=710, sticky=True, recovery_action="NONE")
RETRYABLE = CudaFault(error_code=2, sticky=False, recovery_action="NONE")

QUERY = FaultToleranceRequest(instruction="query_device_fault", params={})


class _ScriptedSentinel(WorkerSentinel):
    """A backend that classifies by a table. `classify_fault` is the method
    a backend overrides, so the record and query logic under test is real."""

    def __init__(self, by_error: dict[BaseException | None, DeviceFault]):
        # The worker is a stand-in because a real `Worker` needs a GPU and
        # model weights.
        worker = SimpleNamespace(
            parallel_config=ParallelConfig(all2all_backend="nixl_ep"),
            vllm_config=SimpleNamespace(speculative_config=None),
        )
        super().__init__(worker)  # type: ignore[arg-type]
        self.by_error = by_error

    def classify_fault(self, exc: BaseException | None) -> DeviceFault | None:
        return self.by_error.get(exc)


def test_recorded_fault_is_handed_to_the_engine_once():
    error = RuntimeError("step failed")
    sentinel = _ScriptedSentinel({error: RETRYABLE})

    sentinel.record_fault(error)

    assert sentinel.handle_command(QUERY) == {"device_fault": RETRYABLE}
    # A fault from before a recovery must not be reported for the next one.
    assert sentinel.handle_command(QUERY) == {"device_fault": None}


def test_error_that_is_not_a_device_fault_records_nothing():
    sentinel = _ScriptedSentinel({})

    sentinel.record_fault(ValueError("bad input"))

    assert sentinel.handle_command(QUERY) == {"device_fault": None}


def test_later_error_does_not_hide_a_fault_that_requires_a_restart():
    first, second = RuntimeError("first"), RuntimeError("second")
    sentinel = _ScriptedSentinel({first: STICKY, second: RETRYABLE})

    sentinel.record_fault(first)
    sentinel.record_fault(second)

    assert sentinel.handle_command(QUERY) == {"device_fault": STICKY}


def test_fault_that_requires_a_restart_replaces_a_retryable_one():
    first, second = RuntimeError("first"), RuntimeError("second")
    sentinel = _ScriptedSentinel({first: RETRYABLE, second: STICKY})

    sentinel.record_fault(first)
    sentinel.record_fault(second)

    assert sentinel.handle_command(QUERY) == {"device_fault": STICKY}


def test_worker_that_recorded_nothing_checks_its_device_when_asked():
    """The device may hold an error that no call has raised yet. `None` is
    how the sentinel asks the backend for it."""
    sentinel = _ScriptedSentinel({None: STICKY})

    assert sentinel.handle_command(QUERY) == {"device_fault": STICKY}
