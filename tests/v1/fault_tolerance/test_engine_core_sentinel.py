# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for EngineCoreSentinel fault handling."""

import queue
import threading
from collections import deque
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm.distributed.kv_transfer.kv_connector.utils import KVConnectorOutput
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.engine import EngineStatusType
from vllm.v1.fault_tolerance.engine_core_sentinel import EngineCoreSentinel
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu_fault import GpuFault


def _model_output(kv_connector_output: KVConnectorOutput | None) -> ModelRunnerOutput:
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
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


def _make_faulting_sentinel(gpu_faults: list[GpuFault]):
    executor = SimpleNamespace(
        is_failed=False,
        classify_fault=lambda exc: gpu_faults,
        collective_rpc=MagicMock(return_value=[{"mask": [0, 1]}]),
    )
    engine = SimpleNamespace(
        scheduler=MagicMock(),
        _send_abort_outputs=MagicMock(),
        model_executor=executor,
        output_queue=queue.Queue(),
    )
    sentinel = EngineCoreSentinel.__new__(EngineCoreSentinel)
    sentinel.engine = engine
    sentinel.engine_index = 0
    sentinel.engine_recovery_timeout_sec = 1
    sentinel.resumed = threading.Event()
    sentinel.status_type = EngineStatusType.HEALTHY
    sentinel.fault_info = None
    sentinel.gpu_fault = None
    return sentinel, engine


def _last_status(engine) -> dict:
    payload = None
    while not engine.output_queue.empty():
        _, outputs = engine.output_queue.get_nowait()
        payload = outputs.utility_output.result.result
    assert payload is not None
    return payload


@pytest.mark.parametrize(
    "fault",
    [
        GpuFault(cuda_error=700, sticky=True, gpu_uuid="GPU-0", recovery_action="NONE"),
        GpuFault(cuda_error=999, sticky=False, recovery_action="GPU_RESET"),
    ],
)
def test_on_fault_restart_required_gpu_fault_is_dead(fault):
    """A fault that retry cannot fix must report DEAD, so the orchestrator
    removes or restarts the rank instead of retrying it."""
    sentinel, engine = _make_faulting_sentinel([fault])

    sentinel.on_fault(RuntimeError("One or more workers failed"))

    assert sentinel.status_type == EngineStatusType.DEAD
    status = _last_status(engine)
    assert status["status"] == "dead"
    assert "restart required" in status["fault_info"]
    assert status["gpu_fault"] == fault.to_dict()
    assert status["gpu_fault"]["restart_required"]
    engine.model_executor.collective_rpc.assert_not_called()


def test_on_fault_without_gpu_fault_stays_retryable():
    sentinel, engine = _make_faulting_sentinel([])

    sentinel.on_fault(RuntimeError("One or more workers failed"))

    assert sentinel.status_type == EngineStatusType.UNHEALTHY
    status = _last_status(engine)
    assert status == {
        "id": 0,
        "status": "unhealthy",
        "fault_info": "RuntimeError",
        "mask": [0, 1],
    }
