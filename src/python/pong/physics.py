"""
Pong physics + AI as pure functions — single source of truth.

Used by:
  - pong/rl/envs/pong.py          (RL training)
  - pong/rl/agents/rule_based.py   (rule-based agent)
  - pong/onnx_modules.py           (ONNX export wrappers)

Symmetric: left and right paddles use identical physics.
All functions are pure tensor ops, torch.where for branching.
"""

from typing import Union

import torch
from torch import Tensor

_Scalar = Union[float, Tensor]

COURT_W = 800.0
COURT_H = 600.0
PADDLE_W = 14.0
PADDLE_H = 110.0
PADDLE_INSET = 26.0
BALL_R = 10.0
BALL_BASE_SPEED = 560.0
PADDLE_SPEED = 480.0
MAX_SCORE = 11
DT = 1.0 / 60.0


def ball_move(ball_x: Tensor, ball_y: Tensor,
              ball_vx: Tensor, ball_vy: Tensor,
              dt: _Scalar = DT) -> tuple[Tensor, Tensor]:
    return ball_x + ball_vx * dt, ball_y + ball_vy * dt


def wall_collide(by: Tensor, bvy: Tensor,
                 h: _Scalar = COURT_H) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    hit_top = by - BALL_R <= 0.0
    by = torch.where(hit_top, torch.tensor(BALL_R), by)
    bvy = torch.where(hit_top, torch.abs(bvy), bvy)

    hit_bottom = by + BALL_R >= h
    by = torch.where(hit_bottom, torch.tensor(h - BALL_R), by)
    bvy = torch.where(hit_bottom, -torch.abs(bvy), bvy)

    return by, bvy, hit_top, hit_bottom


def paddle_collide(bx: Tensor, by: Tensor, bvx: Tensor, bvy: Tensor,
                   paddle_y: Tensor, paddle_x: _Scalar, rally: Tensor,
                   going_left: bool,
                   ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Symmetric paddle collision.

    going_left=True  → left paddle  (ball moving left,  deflect rightward)
    going_left=False → right paddle (ball moving right, deflect leftward)
    """
    if going_left:
        hit = (
            (bvx < 0)
            & ((bx - BALL_R) <= (paddle_x + PADDLE_W / 2.0))
            & (by >= paddle_y - PADDLE_H / 2.0)
            & (by <= paddle_y + PADDLE_H / 2.0)
        )
    else:
        hit = (
            (bvx > 0)
            & ((bx + BALL_R) >= (paddle_x - PADDLE_W / 2.0))
            & (by >= paddle_y - PADDLE_H / 2.0)
            & (by <= paddle_y + PADDLE_H / 2.0)
        )

    n = (by - paddle_y) / (PADDLE_H / 2.0)
    angle = n * 0.95
    new_rally = rally + 1.0
    speed = BALL_BASE_SPEED + new_rally * 15.0

    raw_vx = speed * torch.cos(angle)
    raw_vy = speed * torch.sin(angle)
    mag = torch.sqrt(raw_vx * raw_vx + raw_vy * raw_vy).clamp(min=1.0)
    norm_vx = (raw_vx / mag) * speed
    norm_vy = (raw_vy / mag) * speed

    if going_left:
        new_bx = torch.where(hit, torch.tensor(paddle_x + PADDLE_W / 2.0 + BALL_R), bx)
        new_bvx = torch.where(hit, torch.abs(norm_vx), bvx)
    else:
        new_bx = torch.where(hit, torch.tensor(paddle_x - PADDLE_W / 2.0 - BALL_R), bx)
        new_bvx = torch.where(hit, -torch.abs(norm_vx), bvx)

    new_bvy = torch.where(hit, norm_vy, bvy)
    new_rally = torch.where(hit, new_rally, rally)

    return new_bx, new_bvx, new_bvy, new_rally, hit


def score_detect(bx: Tensor, w: float = COURT_W) -> tuple[Tensor, Tensor]:
    scored_left = bx > w + 60.0
    scored_right = bx < -60.0
    return scored_left, scored_right


def apply_action(paddle_y: Tensor, action: Tensor,
                 court_h: _Scalar = COURT_H, dt: _Scalar = DT) -> Tensor:
    # action: float32 — 0.0=NOOP, 1.0=DOWN(Y+), 2.0=UP(Y-) (rounded internally)
    a = torch.round(action)
    move = (a == 1).float() - (a == 2).float()
    return torch.clamp(
        paddle_y + move * PADDLE_SPEED * dt,
        PADDLE_H / 2.0,
        court_h - PADDLE_H / 2.0,
    )


def ai_track(ball_y: Tensor, ball_vy: Tensor,
             memory_y: Tensor,
             reaction: float, jitter: float, look_ahead: float,
             rand_val: Tensor, court_h: _Scalar = COURT_H,
             dt: _Scalar = DT,
             ) -> Tensor:
    """Core AI tracking — returns new_memory_y (smoothed target position)."""
    target = ball_y + ball_vy * look_ahead + rand_val * jitter
    target = torch.clamp(target, 60.0, court_h - 60.0)

    t = 1.0 - torch.pow(torch.tensor(0.0006), dt / reaction)
    return memory_y + (target - memory_y) * t


def target_to_action(target_y: Tensor, own_y: Tensor,
                     threshold: float = 8.0) -> Tensor:
    """Convert continuous target_y to discrete action (float32): 0=NOOP, 1=DOWN(Y+), 2=UP(Y-)."""
    diff = target_y - own_y
    return torch.where(
        diff > threshold,
        torch.tensor(1.0),
        torch.where(
            diff < -threshold,
            torch.tensor(2.0),
            torch.tensor(0.0),
        ),
    )
