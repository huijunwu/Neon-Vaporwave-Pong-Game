from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

from pong.rl.types import Env
from pong.rl.functional import split_seed, manual_uniform
from pong.physics import (
    COURT_W, COURT_H, PADDLE_H, PADDLE_INSET, BALL_BASE_SPEED,
    MAX_SCORE, DT,
    ball_move, wall_collide, paddle_collide, score_detect, apply_action,
)
from pong.rl.types import Discrete
from pong.rl.types import Timestep


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


def _serve_ball(seed: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    s1, s2, s3 = split_seed(seed, 3)
    angle_rand = manual_uniform(s1)
    dir_rand = manual_uniform(s2)

    angle = (angle_rand - 0.5) * 1.2
    direction = torch.where(dir_rand > 0.5, torch.tensor(1.0), torch.tensor(-1.0))

    bvx = direction * BALL_BASE_SPEED * torch.cos(angle)
    bvy = BALL_BASE_SPEED * torch.sin(angle)

    return (
        torch.tensor(COURT_W / 2.0),
        torch.tensor(COURT_H / 2.0),
        bvx, bvy, s3,
    )


def _get_obs(state: PongState) -> Tensor:
    """(n_agents=2, obs_dim=6): each agent sees ball + own paddle + opponent paddle."""
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


class PongEnv:

    @property
    def n_agents(self) -> int:
        return 2

    @property
    def obs_space(self) -> Discrete:
        return Discrete(n=6)

    @property
    def action_space(self) -> Discrete:
        return Discrete(n=3)

    def reset(self, seed: Tensor) -> tuple[PongState, Timestep]:
        s1, s2 = split_seed(seed, 2)
        bx, by, bvx, bvy, s_next = _serve_ball(s1)

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
        # actions: (2,) int — 0=NOOP, 1=UP, 2=DOWN
        new_left_y = apply_action(state.paddle_left_y, actions[0])
        new_right_y = apply_action(state.paddle_right_y, actions[1])

        bx, by = ball_move(state.ball_x, state.ball_y, state.ball_vx, state.ball_vy)
        bvx = state.ball_vx
        bvy = state.ball_vy
        rally = state.rally

        by, bvy, hit_top, hit_bottom = wall_collide(by, bvy)

        bx, bvx, bvy, rally, hit_left = paddle_collide(
            bx, by, bvx, bvy, new_left_y, PADDLE_INSET, rally, going_left=True,
        )
        bx, bvx, bvy, rally, hit_right = paddle_collide(
            bx, by, bvx, bvy, new_right_y, COURT_W - PADDLE_INSET, rally, going_left=False,
        )

        scored_left, scored_right = score_detect(bx)
        scored_any = scored_left | scored_right

        new_score_left = state.score_left + scored_left.float()
        new_score_right = state.score_right + scored_right.float()

        serve_bx, serve_by, serve_bvx, serve_bvy, new_seed = _serve_ball(state.seed)
        bx = torch.where(scored_any, serve_bx, bx)
        by = torch.where(scored_any, serve_by, by)
        bvx = torch.where(scored_any, serve_bvx, bvx)
        bvy = torch.where(scored_any, serve_bvy, bvy)
        rally = torch.where(scored_any, torch.tensor(0.0), rally)
        final_seed = torch.where(scored_any, new_seed, state.seed)

        game_over = (new_score_left >= MAX_SCORE) | (new_score_right >= MAX_SCORE)

        new_state = PongState(
            ball_x=bx, ball_y=by, ball_vx=bvx, ball_vy=bvy,
            paddle_left_y=new_left_y, paddle_right_y=new_right_y,
            score_left=new_score_left, score_right=new_score_right,
            rally=rally,
            step_count=state.step_count + 1.0,
            seed=final_seed,
        )

        obs = _get_obs(new_state)

        events = torch.stack([
            hit_left.float(), hit_right.float(),
            hit_top.float(), hit_bottom.float(),
            scored_left.float(), scored_right.float(),
        ])

        return new_state, Timestep(
            obs=obs,
            reward=torch.stack([
                scored_left.float() - scored_right.float(),
                scored_right.float() - scored_left.float(),
            ]),
            done=torch.stack([game_over, game_over]),
            truncated=torch.zeros(2, dtype=torch.bool),
            info=events,
        )

    def reset_done(self, state: PongState, timestep: Timestep, seed: Tensor) -> tuple[PongState, Timestep]:
        from pong.rl.functional import auto_reset
        return auto_reset(self, state, timestep, seed)
