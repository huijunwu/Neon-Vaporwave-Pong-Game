from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from pong.rl.functional import split_seed, manual_uniform
from pong.physics import COURT_H, BALL_BASE_SPEED, DT, ai_track, target_to_action


class RuleBasedState(NamedTuple):
    memory_y: Tensor
    seed: Tensor


class RuleBasedAgent:
    """Wraps ai_track() + target_to_action() as a discrete-action Agent.

    Drop-in replaceable with a neural network Agent.
    """

    def __init__(self, reaction: float = 0.16,
                 jitter: float = 22.0, look_ahead: float = 0.14,
                 threshold: float = 8.0):
        self.reaction = reaction
        self.jitter = jitter
        self.look_ahead = look_ahead
        self.threshold = threshold

    def initial_state(self, seed: Tensor) -> RuleBasedState:
        return RuleBasedState(
            memory_y=torch.tensor(COURT_H / 2.0),
            seed=seed,
        )

    def act(self, obs: Tensor, state: RuleBasedState | None = None) -> tuple[Tensor, RuleBasedState]:
        if state is None:
            state = self.initial_state(torch.tensor(0, dtype=torch.int64))

        # obs: (6,) = [ball_x/W, ball_y/H, ball_vx/SPEED, ball_vy/SPEED, own_y/H, opp_y/H]
        ball_y = obs[1] * COURT_H
        ball_vy = obs[3] * BALL_BASE_SPEED
        own_y = obs[4] * COURT_H

        s1, s2 = split_seed(state.seed)
        rand_val = manual_uniform(s1) * 2.0 - 1.0

        new_memory = ai_track(
            ball_y, ball_vy, state.memory_y,
            self.reaction, self.jitter, self.look_ahead,
            rand_val, DT,
        )

        action = target_to_action(new_memory, own_y, self.threshold)

        return action, RuleBasedState(memory_y=new_memory, seed=s2)
