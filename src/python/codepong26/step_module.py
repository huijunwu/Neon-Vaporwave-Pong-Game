"""
PongStepModule — Pong environment as nn.Module.

  forward() → exported to ONNX as pong_step.onnx (JS calls this)
  reset/step/reset_done → Python-only (RL training)
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from codepong26.physics import (
    COURT_W, COURT_H, BALL_BASE_SPEED,
    full_step, serve_ball_from_seed,
)
from codepong26.functional import Timestep, split_seed, manual_uniform


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


def _get_obs(state: PongState) -> Tensor:
    device = state.ball_x.device
    court_w = torch.tensor(COURT_W, device=device)
    court_h = torch.tensor(COURT_H, device=device)
    ball_speed = torch.tensor(BALL_BASE_SPEED, device=device)
    shared = torch.stack([
        state.ball_x / court_w,
        state.ball_y / court_h,
        state.ball_vx / ball_speed,
        state.ball_vy / ball_speed,
    ])
    court_h_tensor = torch.tensor(COURT_H, device=device)
    left_obs = torch.cat([shared, torch.stack([
        state.paddle_left_y / court_h_tensor,
        state.paddle_right_y / court_h_tensor,
    ])])
    right_obs = torch.cat([shared, torch.stack([
        state.paddle_right_y / court_h_tensor,
        state.paddle_left_y / court_h_tensor,
    ])])
    return torch.stack([left_obs, right_obs])


class PongStepModule(nn.Module):

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
        device = seed.device
        s1, s2 = split_seed(seed, 2)
        bx, by, bvx, bvy, s_next = serve_ball_from_seed(s1, COURT_W, COURT_H)
        bx = bx.to(device)
        by = by.to(device)
        bvx = bvx.to(device)
        bvy = bvy.to(device)
        s_next = s_next.to(device)

        state = PongState(
            ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
            paddle_left_y=torch.tensor(COURT_H / 2.0, device=device),
            paddle_right_y=torch.tensor(COURT_H / 2.0, device=device),
            score_left=torch.tensor(0.0, device=device),
            score_right=torch.tensor(0.0, device=device),
            rally=torch.tensor(0.0, device=device),
            step_count=torch.tensor(0.0, device=device),
            seed=s_next,
        )
        return state, Timestep(
            obs=_get_obs(state),
            reward=torch.zeros(2, device=device),
            done=torch.zeros(2, dtype=torch.bool, device=device),
            truncated=torch.zeros(2, dtype=torch.bool, device=device),
            info=torch.tensor(0.0, device=device),
        )

    def step(self, state: PongState, actions: Tensor) -> tuple[PongState, Timestep]:
        """Python-only: structured step for RL training."""
        device = state.ball_x.device
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
            torch.tensor(COURT_W, device=device), torch.tensor(COURT_H, device=device),
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
            truncated=torch.zeros(2, dtype=torch.bool, device=device),
            info=events,
        )

    def reset_done(self, state: PongState, timestep: Timestep,
                   seed: Tensor) -> tuple[PongState, Timestep]:
        from codepong26.functional import auto_reset
        return auto_reset(self, state, timestep, seed)
