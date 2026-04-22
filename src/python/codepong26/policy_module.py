"""
PongPolicyModule — Rule-based agent as nn.Module.

  forward() → exported to ONNX as pong_policy.onnx (JS calls this)
  act/initial_state → Python-only (RL training)

  Swappable: replace with a neural network policy module.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from codepong26.physics import COURT_H, rule_based_policy
from codepong26.functional import split_seed, manual_uniform


class PolicyState(NamedTuple):
    memory_y: Tensor
    seed: Tensor


class PongPolicyModule(nn.Module):

    def __init__(self, reaction: float = 0.16,
                 jitter: float = 22.0, look_ahead: float = 0.14,
                 threshold: float = 8.0):
        super().__init__()
        self.reaction = reaction
        self.jitter = jitter
        self.look_ahead = look_ahead
        self.threshold = threshold

    def forward(self, obs, memory_y, rand_val, H):
        """ONNX-exported: obs → action + new_memory."""
        return rule_based_policy(
            obs, memory_y, rand_val,
            self.reaction, self.jitter, self.look_ahead, self.threshold,
            court_h=H,
        )

    def initial_state(self, seed: Tensor) -> PolicyState:
        return PolicyState(
            memory_y=torch.tensor(COURT_H / 2.0),
            seed=seed,
        )

    def act(self, obs: Tensor, state: PolicyState | None = None) -> tuple[Tensor, PolicyState]:
        """Python-only: RL-friendly wrapper with seed-based RNG."""
        if state is None:
            state = self.initial_state(torch.tensor(0, dtype=torch.int64))

        s1, s2 = split_seed(state.seed)
        rand_val = manual_uniform(s1) * 2.0 - 1.0

        action, new_memory = self.forward(
            obs, state.memory_y, rand_val, torch.tensor(COURT_H),
        )
        return action, PolicyState(memory_y=new_memory, seed=s2)
