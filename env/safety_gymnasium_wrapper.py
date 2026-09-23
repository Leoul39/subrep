"""
Safety-Gymnasium Environment Wrapper for SubRep.

This wrapper adapts a Safety-Gymnasium environment to the SubRepBaseEnv contract,
exposing a 2D motive vector [Safety, Task] and injecting ``task_payoff`` into
each step's info dict.

The real ``safety_gymnasium`` package requires Python 3.10 and is an optional
dependency.  Tests that need to run without the real package should inject a
fake environment via the ``env`` constructor parameter instead of relying on
``safety_gymnasium.make``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

_SAFETY_GYM_AVAILABLE = False
try:
    import safety_gymnasium  # type: ignore[import]
    _SAFETY_GYM_AVAILABLE = True
except ImportError:
    pass


class SafetyGymnasiumEnv:
    """
    Wraps a Safety-Gymnasium environment to conform to SubRepBaseEnv.

    Motive vector: [Safety, Task]
      - Safety  = 1.0 - cost (clipped to [0, 1])
      - Task    = task reward from the underlying environment

    Args:
        env_id: Safety-Gymnasium environment ID (e.g. ``"SafetyPointGoal1-v0"``).
                Ignored when ``env`` is provided directly.
        env:    Pre-constructed environment instance.  When provided, ``env_id``
                is not used and ``safety_gymnasium`` need not be installed.
        seed:   Optional seed passed to the first ``reset()`` call.
    """

    def __init__(
        self,
        env_id: str = "SafetyPointGoal1-v0",
        env: Any = None,
        seed: Optional[int] = None,
    ) -> None:
        if env is not None:
            self._env = env
        elif _SAFETY_GYM_AVAILABLE:
            self._env = safety_gymnasium.make(env_id)
        else:
            raise ImportError(
                "safety_gymnasium is not installed and no env was injected. "
                "Install safety-gymnasium (requires Python 3.10) or pass a "
                "fake env object via the 'env' parameter."
            )
        self._seed = seed
        if seed is not None:
            self._env.reset(seed=seed)

    # ------------------------------------------------------------------
    # SubRepBaseEnv interface
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> Dict[str, Any]:
        """Environment metadata dictionary conforming to SubRep specification."""
        return {
            "environment_id": "subrep_safety_gymnasium_v1",
            "motive_names": ["Safety", "Task"],
            "motive_schema_version": "1.0",
            "payoff_schema_version": "1.0",
            "observation_schema_version": "1.0",
            "action_schema_version": "1.0",
        }

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment and return (observation, info)."""
        if seed is not None:
            self._seed = seed
            obs, info = self._env.reset(seed=seed)
        else:
            reset_out = self._env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) == 2:
                obs, info = reset_out
            else:
                obs, info = reset_out, {}
        return obs, dict(info) if isinstance(info, dict) else {}

    def step(self, action: Any) -> Tuple[Any, np.ndarray, bool, bool, Dict[str, Any]]:
        """
        Step the environment and return the SubRepBaseEnv 5-tuple.

        Returns:
            (observation, motive_vector, terminated, truncated, info)
            motive_vector shape: (2,) — [Safety, Task]
            info["task_payoff"]: scalar task reward.
        """
        step_out = self._env.step(action)
        if len(step_out) == 5:
            obs, reward, cost, terminated, info = step_out
            truncated = False
        elif len(step_out) == 6:
            obs, reward, cost, terminated, truncated, info = step_out
        else:
            raise ValueError(
                f"Expected Safety-Gymnasium step() to return 5 or 6 elements, got {len(step_out)}"
            )

        info = dict(info) if isinstance(info, dict) else {}

        # Build the 2D motive vector: [Safety, Task]
        # Safety = 1 - cost, clipped to [0, 1] so larger is always better.
        safety = float(np.clip(1.0 - float(cost), 0.0, 1.0))
        task = float(reward)
        motive_vector = np.array([safety, task], dtype=np.float32)

        # Inject scalar task_payoff for SubRepBaseEnv compliance.
        info["task_payoff"] = task
        info["cost"] = float(cost)

        return obs, motive_vector, bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Close and clean up environment resources."""
        self._env.close()
