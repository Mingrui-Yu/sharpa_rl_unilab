"""Runtime protocol for Manager-Based term dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from unilab.managers._types import ManagerBasedRlEnv
    from unilab.managers.action_manager import ActionManager
    from unilab.managers.event_manager import EventManager
    from unilab.managers.observation_manager import ObservationManager
    from unilab.managers.termination_manager import TerminationManager

    class SharpaEnv(ManagerBasedRlEnv, Protocol):
        """Manager fields consumed by Sharpa terms."""

        @property
        def cfg(self) -> Any: ...

        @property
        def common_step_counter(self) -> int: ...

        @property
        def event_manager(self) -> EventManager: ...

        @property
        def action_manager(self) -> ActionManager: ...

        @property
        def observation_manager(self) -> ObservationManager: ...

        @property
        def termination_manager(self) -> TerminationManager: ...

        @property
        def reset_time_outs(self) -> np.ndarray: ...

        @property
        def reset_terminated(self) -> np.ndarray: ...

        @property
        def extras(self) -> dict[str, Any]: ...
