"""
ONNX-first game modules — each class is a complete component:
  forward() → exported to ONNX (flat scalars for JS)
  reset/step/act → Python-only (RL training with structured types)

  PongStepModule  → pong_step.onnx   (env: physics + scoring + serve + game-over)
  PongPolicyModule → pong_policy.onnx (agent: rule-based, swappable for NN)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from pong.physics import (
    COURT_W, COURT_H, BALL_BASE_SPEED,
    full_step, rule_based_policy, serve_ball_from_seed,
)
from pong.functional import Timestep, split_seed, manual_uniform


class PongState(NamedTuple):
    ball_x: Tensor
    ball_y: Tensor
    ball_vx: Tensor
    ball_vy: Tensor
    paddle_left_y: Tensor
    paddle_right_y: Tensor
    score_left: Tensor
    score_right: Tensor
    rally: Tensor
    step_count: Tensor
    seed: Tensor


class PolicyState(NamedTuple):
    memory_y: Tensor
    seed: Tensor


# ── Helpers ──

def _get_obs(state: PongState) -> Tensor:
    shared = torch.stack([
        state.ball_x / COURT_W,
        state.ball_y / COURT_H,
        state.ball_vx / BALL_BASE_SPEED,
        state.ball_vy / BALL_BASE_SPEED,
    ])
    left_obs = torch.cat([shared, torch.stack([
        state.paddle_left_y / COURT_H,
        state.paddle_right_y / COURT_H,
    ])])
    right_obs = torch.cat([shared, torch.stack([
        state.paddle_right_y / COURT_H,
        state.paddle_left_y / COURT_H,
    ])])
    return torch.stack([left_obs, right_obs])


# ═══════════════════════════════════════════════════════════════════
#  PongStepModule — env (ONNX + RL)
# ═══════════════════════════════════════════════════════════════════

class PongStepModule(nn.Module):
    """Pong environment as nn.Module.

    forward() → ONNX export (flat scalars, JS calls this)
    reset/step/reset_done → Python-only (RL training)
    """

    n_agents = 2
    obs_dim = 6
    n_actions = 3

    def forward(self, ball_x, ball_y, ball_vx, ball_vy,
                paddle_left_y, paddle_right_y,
                score_left, score_right,
                rally,
                action_left, action_right,
                rand_angle, rand_dir,
                W, H):
        """ONNX-exported: full game step with flat scalar interface."""
        return full_step(
            ball_x, ball_y, ball_vx, ball_vy,
            paddle_left_y, paddle_right_y,
            score_left, score_right,
            rally,
            action_left, action_right,
            rand_angle, rand_dir,
            court_w=W, court_h=H,
        )

    def reset(self, seed: Tensor) -> tuple[PongState, Timestep]:
        s1, s2 = split_seed(seed, 2)
        bx, by, bvx, bvy, s_next = serve_ball_from_seed(s1, COURT_W, COURT_H)

        state = PongState(
            ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
            paddle_left_y=torch.tensor(COURT_H / 2.0),
            paddle_right_y=torch.tensor(COURT_H / 2.0),
            score_left=torch.tensor(0.0),
            score_right=torch.tensor(0.0),
            rally=torch.tensor(0.0),
            step_count=torch.tensor(0.0),
            seed=s_next,
        )
        return state, Timestep(
            obs=_get_obs(state),
            reward=torch.zeros(2),
            done=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
            info=torch.tensor(0.0),
        )

    def step(self, state: PongState, actions: Tensor) -> tuple[PongState, Timestep]:
        """Python-only: structured step for RL training."""
        s1, s2, s_next = split_seed(state.seed, 3)
        rand_angle = manual_uniform(s1)
        rand_dir = manual_uniform(s2)

        (bx, by, bvx, bvy,
         new_left_y, new_right_y,
         new_score_left, new_score_right,
         new_rally,
         events, game_over) = self.forward(
            state.ball_x, state.ball_y, state.ball_vx, state.ball_vy,
            state.paddle_left_y, state.paddle_right_y,
            state.score_left, state.score_right,
            state.rally,
            actions[0], actions[1],
            rand_angle, rand_dir,
            torch.tensor(COURT_W), torch.tensor(COURT_H),
        )

        scored_any = (events[4] > 0.5) | (events[5] > 0.5)
        new_seed = torch.where(scored_any, s_next, state.seed)

        new_state = PongState(
            ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
            paddle_left_y=new_left_y, paddle_right_y=new_right_y,
            score_left=new_score_left, score_right=new_score_right,
            rally=new_rally,
            step_count=state.step_count + 1.0,
            seed=new_seed,
        )

        scored_left = events[4] > 0.5
        scored_right = events[5] > 0.5

        return new_state, Timestep(
            obs=_get_obs(new_state),
            reward=torch.stack([
                scored_left.float() - scored_right.float(),
                scored_right.float() - scored_left.float(),
            ]),
            done=torch.stack([game_over > 0.5, game_over > 0.5]),
            truncated=torch.zeros(2, dtype=torch.bool),
            info=events,
        )

    def reset_done(self, state: PongState, timestep: Timestep,
                   seed: Tensor) -> tuple[PongState, Timestep]:
        from pong.functional import auto_reset
        return auto_reset(self, state, timestep, seed)


# ═══════════════════════════════════════════════════════════════════
#  PongPolicyModule — agent (ONNX + RL)
# ═══════════════════════════════════════════════════════════════════

class PongPolicyModule(nn.Module):
    """Rule-based agent as nn.Module.

    forward() → ONNX export (flat scalars, JS calls this)
    act/initial_state → Python-only (RL training)
    """

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
