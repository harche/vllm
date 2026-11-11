# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EngineCoreSentinel fault handling."""

import queue
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.config import ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.utils import KVConnectorOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.fault_tolerance.cuda_fault import CudaFault
from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
from vllm.v1.outputs import ModelRunnerOutput


def _model_output(kv_connector_output: KVConnectorOutput | None) -> ModelRunnerOutput:
    output = ModelRunnerOutput(req_ids=[])
    output.kv_connector_output = kv_connector_output
    return output


def _done_future(result=None, exception=None) -> Future:
    future: Future = Future()
    if exception is not None:
        future.set_exception(exception)
    else:
        future.set_result(result)
    return future


def _make_sentinel(batch_queue: deque) -> tuple[EngineCoreSentinel, MagicMock]:
    scheduler = MagicMock()
    sentinel = EngineCoreSentinel.__new__(EngineCoreSentinel)
    sentinel.engine = SimpleNamespace(batch_queue=batch_queue, scheduler=scheduler)
    return sentinel, scheduler


def test_drain_batch_queue_surfaces_kv_connector_output():
    """Pre-fault KV output must reach the scheduler, not be lost on clear."""
    kv_output = KVConnectorOutput(finished_sending={"req1"})
    batch_queue = deque([(_done_future(_model_output(kv_output)), None, None)])
    sentinel, scheduler = _make_sentinel(batch_queue)

    sentinel._drain_batch_queue()

    assert not batch_queue
    scheduler.update_from_output.assert_called_once()
    sched_output, model_output = scheduler.update_from_output.call_args[0]
    assert isinstance(sched_output, SchedulerOutput)
    assert sched_output.num_scheduled_tokens == {}
    assert model_output.kv_connector_output is kv_output


def test_drain_batch_queue_drops_failures_and_empty_outputs():
    """Failed futures and empty KV outputs are dropped without scheduler calls."""
    batch_queue = deque(
        [
            (_done_future(exception=RuntimeError("boom")), None, None),
            (_done_future(_model_output(None)), None, None),
            (_done_future(_model_output(KVConnectorOutput())), None, None),
        ]
    )
    sentinel, scheduler = _make_sentinel(batch_queue)

    sentinel._drain_batch_queue()

    assert not batch_queue
    scheduler.update_from_output.assert_not_called()


class _StubExecutor:
    """Stands in for the engine's executor. It answers the two commands the
    sentinel sends to the workers while it handles a fault, with one reply
    per worker."""

    def __init__(self, is_failed: bool, worker_faults: list[CudaFault | None]):
        self.is_failed = is_failed
        self.worker_faults = worker_faults
        self.instructions: list[str] = []

    def collective_rpc(self, method, args=(), timeout=None):
        assert method == "handle_ft_command"
        instruction = args[0].instruction
        self.instructions.append(instruction)
        if instruction == "query_device_fault":
            if self.worker_faults is _NO_ANSWER:
                raise TimeoutError("RPC call to handle_ft_command timed out.")
            return [{"device_fault": fault} for fault in self.worker_faults]
        return [{"mask": [0, 0]} for _ in self.worker_faults]


# The workers do not answer the device fault query.
_NO_ANSWER: list[CudaFault | None] = []


def _make_fault_sentinel(
    worker_faults: list[CudaFault | None],
    executor_failed: bool = False,
) -> tuple[EngineCoreSentinel, Callable[[], list[dict]]]:
    """A sentinel built by its real constructor around a stand-in engine, and
    a reader for the statuses it pushes to the client. `worker_faults` is
    what each worker recorded, in rank order.

    The engine is a stand-in because a real `EngineCoreProc` needs model
    weights and a GPU per rank. `test_fault_tolerance_e2e.py` covers the same
    behavior through a real server.
    """
    output_queue: queue.Queue = queue.Queue()
    engine = SimpleNamespace(
        engine_index=0,
        scheduler=MagicMock(),
        model_executor=_StubExecutor(executor_failed, worker_faults),
        output_queue=output_queue,
        _send_abort_outputs=lambda aborted: None,
    )
    sentinel = EngineCoreSentinel(engine, ParallelConfig())

    def pushed_statuses() -> list[dict]:
        statuses = []
        while not output_queue.empty():
            _, outputs = output_queue.get_nowait()
            statuses.append(outputs.utility_output.result.result)
        return statuses

    return sentinel, pushed_statuses


STICKY = CudaFault(error_code=710, sticky=True, recovery_action="NONE")
RETRYABLE = CudaFault(error_code=2, sticky=False, recovery_action="NONE")


def test_fault_that_requires_a_restart_is_reported_dead():
    """Recovery in place cannot succeed, so the engine must not offer retry."""
    sentinel, pushed_statuses = _make_fault_sentinel([STICKY])

    sentinel.on_fault(RuntimeError("One or more workers failed"))

    assert pushed_statuses() == [
        {"id": 0, "status": "diagnosing"},
        {
            "id": 0,
            "status": "dead",
            "fault_info": f"RuntimeError: {STICKY}; restart required",
            "device_fault": STICKY.to_dict(),
        },
    ]
    # The mask is of no use to an engine that cannot recover.
    assert sentinel.engine.model_executor.instructions == ["query_device_fault"]


def test_fault_on_a_worker_that_did_not_reply_to_the_step_is_reported():
    """Only one rank replies to a model step. A device fault on another
    rank reaches the engine because every worker is asked."""
    sentinel, pushed_statuses = _make_fault_sentinel([None, RETRYABLE, STICKY, None])

    sentinel.on_fault(TimeoutError("RPC call to execute_model timed out."))

    status = pushed_statuses()[-1]
    assert status["status"] == "dead"
    assert status["device_fault"] == STICKY.to_dict()


def test_retryable_device_fault_is_reported_unhealthy_with_its_classification():
    sentinel, pushed_statuses = _make_fault_sentinel([RETRYABLE])

    sentinel.on_fault(RuntimeError("One or more workers failed"))

    assert pushed_statuses() == [
        {"id": 0, "status": "diagnosing"},
        {
            "id": 0,
            "status": "unhealthy",
            "fault_info": f"RuntimeError: {RETRYABLE}",
            "device_fault": RETRYABLE.to_dict(),
            "mask": [0, 0],
        },
    ]


def test_fault_that_is_not_a_device_fault_is_reported_without_a_classification():
    sentinel, pushed_statuses = _make_fault_sentinel([None, None])

    sentinel.on_fault(RuntimeError("boom"))

    assert pushed_statuses() == [
        {"id": 0, "status": "diagnosing"},
        {"id": 0, "status": "unhealthy", "fault_info": "RuntimeError", "mask": [0, 0]},
    ]


def test_fault_is_reported_unclassified_when_the_workers_do_not_answer():
    sentinel, pushed_statuses = _make_fault_sentinel(_NO_ANSWER)

    sentinel.on_fault(RuntimeError("boom"))

    status = pushed_statuses()[-1]
    assert status["status"] == "unhealthy"
    assert status["fault_info"] == "RuntimeError"
    assert "device_fault" not in status


def test_classification_of_an_earlier_fault_is_not_reported_again():
    sentinel, pushed_statuses = _make_fault_sentinel([RETRYABLE])
    sentinel.on_fault(RuntimeError("first"))
    pushed_statuses()

    sentinel.engine.model_executor.worker_faults = [None]
    sentinel.on_fault(RuntimeError("second"))

    assert "device_fault" not in pushed_statuses()[-1]


def test_failed_executor_is_reported_dead_without_asking_the_workers():
    sentinel, pushed_statuses = _make_fault_sentinel([None], executor_failed=True)

    sentinel.on_fault(RuntimeError("boom"))

    assert pushed_statuses() == [
        {"id": 0, "status": "dead", "fault_info": "RuntimeError"}
    ]
    assert sentinel.engine.model_executor.instructions == []
