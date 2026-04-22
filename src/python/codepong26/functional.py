from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor


class Timestep(NamedTuple):
    obs: Tensor          # (n_agents, *obs_shape)
    reward: Tensor       # (n_agents,)
    done: Tensor         # (n_agents,)  — bool
    truncated: Tensor    # (n_agents,)  — bool
    info: Tensor         # events or scalar 0


def split_seed(seed: Tensor, n: int = 2) -> tuple[Tensor, ...]:
    """vmap-safe RNG: derive n child seeds from a parent seed via simple hashing."""
    seeds = []
    for i in range(n):
        child = (seed * 2654435761 + i) % (2**31)
        seeds.append(child)
    return tuple(seeds)


def manual_uniform(seed: Tensor, shape: tuple[int, ...] = ()) -> Tensor:
    """Deterministic pseudo-random uniform [0,1) from an int64 seed tensor.

    Uses a simple LCG. Good enough for jitter/noise in game envs.
    NOT cryptographically secure. vmap-safe because it's pure tensor math.
    """
    s = seed.to(torch.int64)
    flat_n = 1
    for d in shape:
        flat_n *= d
    if flat_n == 0:
        return torch.zeros(shape)
    values = []
    for _ in range(flat_n):
        s = (s * 6364136223846793005 + 1442695040888963407) % (2**63)
        values.append(s)
    t = torch.stack(values).float() / (2**63)
    return t.reshape(shape) if shape else t.squeeze()


def auto_reset(
    env,
    state,
    timestep: Timestep,
    seed: Tensor,
) -> tuple:
    """Default reset_done: if ANY agent is done, reset the whole env.

    Uses torch.where to stay vmap-friendly (no Python branching on tensor values).
    """
    any_done = timestep.done.any()
    fresh_state, fresh_ts = env.reset(seed)

    def _select(fresh: Tensor, current: Tensor) -> Tensor:
        return torch.where(any_done, fresh, current)

    new_state_fields = tuple(_select(f, c) for f, c in zip(fresh_state, state))
    new_state = type(state)(*new_state_fields)

    new_ts = Timestep(
        obs=_select(fresh_ts.obs, timestep.obs),
        reward=timestep.reward,
        done=_select(fresh_ts.done, timestep.done),
        truncated=_select(fresh_ts.truncated, timestep.truncated),
        info=timestep.info,
    )
    return new_state, new_ts
