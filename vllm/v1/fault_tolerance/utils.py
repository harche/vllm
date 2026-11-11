# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

import msgspec

# Instructions accepted from external callers. The API router and the engine
# both validate against this; keep it the single source of truth so a new
# instruction cannot be allowed at one layer and rejected at the other.
ALLOWED_FT_INSTRUCTIONS = frozenset({"retry", "scale_down"})


class FaultToleranceResult(msgspec.Struct):
    request_id: str
    success: bool
    reason: str | None = None


class FaultToleranceRequest(msgspec.Struct):
    instruction: str
    params: dict[str, Any]
    request_id: str = ""


def exception_chain(exc: BaseException) -> Iterator[BaseException]:
    """`exc`, then each exception it was raised from or while handling."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        yield cur
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__


@dataclass(frozen=True)
class DeviceFault(ABC):
    """A worker's classification of a device fault.

    Each accelerator backend subclasses this, sets `backend`, and may add
    fields. `error_code` is defined by the backend, as are the values of
    `recovery_action`. The engine reads only `restart_required` and passes
    `to_dict()` through to the status API, which can be reached without an API
    key. Keep hardware identifiers such as a device UUID out of the fields.
    """

    backend: ClassVar[str]

    error_code: int
    sticky: bool
    recovery_action: str | None = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        # Reject an incomplete backend when it is imported, not when it
        # reports its first fault.
        super().__init_subclass__(**kwargs)
        if not isinstance(getattr(cls, "backend", None), str):
            raise TypeError(f"{cls.__name__} must set `backend`")
        if cls.restart_required is DeviceFault.restart_required:
            raise TypeError(f"{cls.__name__} must define `restart_required`")

    @property
    @abstractmethod
    def restart_required(self) -> bool:
        """Whether the process must restart before it can use the device."""

    def to_dict(self) -> dict[str, Any]:
        return (
            {"backend": self.backend}
            | asdict(self)
            | {"restart_required": self.restart_required}
        )

    def __str__(self) -> str:
        return (
            f"device fault: backend={self.backend} error_code={self.error_code} "
            f"sticky={self.sticky} recovery_action={self.recovery_action}"
        )


def select_device_fault(faults: list[DeviceFault]) -> DeviceFault | None:
    """Pick the fault an engine reports from its workers' classifications.

    A fault that requires a restart takes precedence, because it decides the
    engine's status.
    """
    for fault in faults:
        if fault.restart_required:
            return fault
    return faults[0] if faults else None


def describe_fault(exc: BaseException, fault: DeviceFault | None) -> str:
    """Build `fault_info` from typed fields only, never from the exception
    message, which may hold paths or configuration.

    The text is for people and starts with the class name of `exc`. A client
    that decides what to do reads `status` and `device_fault` instead.
    """
    info = type(exc).__name__
    if fault is not None:
        info += f": {fault}"
        if fault.restart_required:
            info += "; restart required"
    return info
