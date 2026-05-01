from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

from codepong26.physics import (
    COURT_W, COURT_H, BALL_BASE_SPEED, PADDLE_SPEED,
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
    paddle_left_vy: Tensor
    paddle_right_vy: Tensor
    score_left: Tensor
    score_right: Tensor
    rally: Tensor
    step_count: Tensor
    seed: Tensor


def _get_obs(state: PongState) -> Tensor:
    device = state.ball_x.device
    court_w    = torch.tensor(COURT_W,        device=device)
    court_h    = torch.tensor(COURT_H,        device=device)
    ball_speed = torch.tensor(BALL_BASE_SPEED, device=device)
    pad_speed  = torch.tensor(PADDLE_SPEED,   device=device)

    shared = torch.stack([
        state.ball_x  / court_w,
        state.ball_y  / court_h,
        state.ball_vx / ball_speed,
        state.ball_vy / ball_speed,
    ])
    shared_mirrored = torch.stack([
        (court_w - state.ball_x) / court_w,
        state.ball_y  / court_h,
        -state.ball_vx / ball_speed,
        state.ball_vy  / ball_speed,
    ])

    left_obs = torch.cat([shared, torch.stack([
        state.paddle_left_y  / court_h,
        state.paddle_right_y / court_h,
        state.paddle_left_vy  / pad_speed,
        state.paddle_right_vy / pad_speed,
    ])])
    right_obs = torch.cat([shared_mirrored, torch.stack([
        state.paddle_right_y / court_h,
        state.paddle_left_y  / court_h,
        state.paddle_right_vy / pad_speed,
        state.paddle_left_vy  / pad_speed,
    ])])
    return torch.stack([left_obs, right_obs])


class PongStepModule(nn.Module):

    n_agents     = 2
    obs_dim      = 8
    action_dim   = 1
    n_dividers   = 5

    EV_HIT_LEFT       = 0
    EV_HIT_RIGHT      = 1
    EV_WALL_TOP       = 2
    EV_WALL_BOTTOM    = 3
    EV_SCORED_LEFT    = 4
    EV_SCORED_RIGHT   = 5
    EV_CROSSED_ZONE   = 6

    def forward(self, ball_x, ball_y, ball_vx, ball_vy,
                paddle_left_y, paddle_right_y,
                score_left, score_right,
                rally,
                action_left, action_right,
                rand_angle, rand_dir,
                W, H):
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
        zero = torch.tensor(0.0, device=device)
        state = PongState(
            ball_x=bx.to(device), ball_y=by.to(device),
            ball_vx=bvx.to(device), ball_vy=bvy.to(device),
            paddle_left_y=torch.tensor(COURT_H / 2.0, device=device),
            paddle_right_y=torch.tensor(COURT_H / 2.0, device=device),
            paddle_left_vy=zero,
            paddle_right_vy=zero,
            score_left=zero, score_right=zero,
            rally=zero, step_count=zero,
            seed=s_next.to(device),
        )
        return state, Timestep(
            obs=_get_obs(state),
            reward=torch.zeros(2, device=device),
            done=torch.zeros(2, dtype=torch.bool, device=device),
            truncated=torch.zeros(2, dtype=torch.bool, device=device),
            info=torch.tensor(0.0, device=device),
        )

    def step(self, state: PongState, actions: Tensor) -> tuple[PongState, Timestep]:
        device = state.ball_x.device
        s1, s2, s_next = split_seed(state.seed, 3)
        rand_angle = manual_uniform(s1)
        rand_dir   = manual_uniform(s2)

        (bx, by, bvx, bvy,
         new_left_y,  new_right_y,
         new_left_vy, new_right_vy,
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

        crossed_zone = torch.zeros((), dtype=torch.bool, device=device)
        for k in range(1, self.n_dividers + 1):
            divider = COURT_W * k / (self.n_dividers + 1)
            crossed_zone = crossed_zone | (((state.ball_x - divider) * (bx - divider)) <= 0.0)

        events = torch.stack([
            events[0], events[1], events[2], events[3], events[4], events[5],
            crossed_zone.float(),
        ])

        scored_any = (events[4] > 0.5) | (events[5] > 0.5)
        new_seed   = torch.where(scored_any, s_next, state.seed)

        new_state = PongState(
            ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
            paddle_left_y=new_left_y,   paddle_right_y=new_right_y,
            paddle_left_vy=new_left_vy, paddle_right_vy=new_right_vy,
            score_left=new_score_left,  score_right=new_score_right,
            rally=new_rally,
            step_count=state.step_count + 1.0,
            seed=new_seed,
        )

        scored_left  = events[4] > 0.5
        scored_right = events[5] > 0.5

        return new_state, Timestep(
            obs=_get_obs(new_state),
            reward=torch.stack([
                scored_left.float()  - scored_right.float(),
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
