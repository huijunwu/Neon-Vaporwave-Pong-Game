from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

from torch import Tensor


# ── Spaces ──

class Discrete(NamedTuple):
    n: int


class Box(NamedTuple):
    low: Tensor
    high: Tensor
    shape: tuple[int, ...]


Space = Discrete | Box


# ── Timestep ──

class Timestep(NamedTuple):
    obs: Tensor          # (n_agents, *obs_shape)
    reward: Tensor       # (n_agents,)
    done: Tensor         # (n_agents,)  — bool
    truncated: Tensor    # (n_agents,)  — bool
    info: Tensor         # (n_agents, *info_shape) or scalar 0 if unused


# ── Agent Protocol ──

@runtime_checkable
class Agent(Protocol):
    """Agent interface — same for rule-based and neural network policies.

    act() takes an observation and returns a discrete action.
    Swap RuleBasedAgent for NNAgent without changing any other code.
    """

    def act(self, obs: Tensor, state: Any | None = None) -> tuple[Tensor, Any | None]:
        """
        obs:   (*obs_shape) — single agent's observation.
        state: optional agent-internal state (RNN hidden, rule-based memory).

        Returns: (action, new_state)
          action:    scalar int64 tensor for Discrete.
          new_state: updated agent state (or None if stateless).
        """
        ...


# ── Env Protocol ──

@runtime_checkable
class Env(Protocol):
    """
    Functional RL environment for torch.compile + torch.vmap.

    Contract:
      - All methods are pure functions: no self mutation, no side effects.
      - State and Timestep are NamedTuples of Tensors with fixed shapes.
      - torch.where instead of Python if/else on tensor values.
      - n_agents >= 1. Single-agent envs use n_agents=1.

    Shapes (before vmap):
      obs:       (n_agents, *obs_shape)
      reward:    (n_agents,)
      done:      (n_agents,)
      actions:   (n_agents,) for Discrete / (n_agents, *act_shape) for Box
    """

    @property
    def n_agents(self) -> int: ...

    @property
    def obs_space(self) -> Space: ...

    @property
    def action_space(self) -> Space: ...

    def reset(self, seed: Tensor) -> tuple[Any, Timestep]:
        """seed: int64 scalar tensor. Returns: (env_state, timestep)."""
        ...

    def step(self, state: Any, actions: Tensor) -> tuple[Any, Timestep]:
        """Pure functional step. Returns: (new_state, timestep)."""
        ...

    def reset_done(self, state: Any, timestep: Timestep, seed: Tensor) -> tuple[Any, Timestep]:
        """Auto-reset done envs in a vmap batch using torch.where."""
        ...
