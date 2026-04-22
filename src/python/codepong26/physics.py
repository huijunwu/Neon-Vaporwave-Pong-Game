"""
Pong physics + AI as pure functions — single source of truth.

Used by:
  - pong/onnx_modules.py  (PongStepModule + PongPolicyModule)

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


def serve_ball_from_rand(rand_angle: Tensor, rand_dir: Tensor,
                         court_w: _Scalar = COURT_W,
                         court_h: _Scalar = COURT_H,
                         ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Ball serve from random values (provided by caller).
    rand_angle: uniform [0,1), rand_dir: uniform [0,1).
    Returns (bx, by, bvx, bvy)."""
    angle = (rand_angle - 0.5) * 1.2
    direction = torch.where(rand_dir > 0.5, torch.tensor(1.0), torch.tensor(-1.0))
    bvx = direction * BALL_BASE_SPEED * torch.cos(angle)
    bvy = BALL_BASE_SPEED * torch.sin(angle)
    cx = court_w / 2.0 if isinstance(court_w, Tensor) else torch.tensor(court_w / 2.0)
    cy = court_h / 2.0 if isinstance(court_h, Tensor) else torch.tensor(court_h / 2.0)
    return cx, cy, bvx, bvy


def serve_ball_from_seed(seed: Tensor, court_w: _Scalar = COURT_W,
                         court_h: _Scalar = COURT_H,
                         ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Deterministic ball serve from a seed (for RL training).
    Returns (bx, by, bvx, bvy, new_seed)."""
    from codepong26.functional import split_seed, manual_uniform
    s1, s2, s3 = split_seed(seed, 3)
    bx, by, bvx, bvy = serve_ball_from_rand(
        manual_uniform(s1), manual_uniform(s2), court_w, court_h,
    )
    return bx, by, bvx, bvy, s3


def full_step(ball_x: Tensor, ball_y: Tensor, ball_vx: Tensor, ball_vy: Tensor,
              paddle_left_y: Tensor, paddle_right_y: Tensor,
              score_left: Tensor, score_right: Tensor,
              rally: Tensor,
              action_left: Tensor, action_right: Tensor,
              rand_angle: Tensor, rand_dir: Tensor,
              court_w: _Scalar = COURT_W, court_h: _Scalar = COURT_H,
              ) -> tuple:
    """Complete game step: physics + scoring + auto-serve + game-over.

    rand_angle, rand_dir: uniform [0,1) from caller (JS or RL seed-based).
    Used for serve direction when a point is scored.

    Returns: (new_ball_x, new_ball_y, new_ball_vx, new_ball_vy,
              new_paddle_left_y, new_paddle_right_y,
              new_score_left, new_score_right,
              new_rally,
              events[6], game_over)
    """
    new_left_y = apply_action(paddle_left_y, action_left, court_h=court_h)
    new_right_y = apply_action(paddle_right_y, action_right, court_h=court_h)

    bx, by = ball_move(ball_x, ball_y, ball_vx, ball_vy)
    bvx = ball_vx
    bvy = ball_vy

    by, bvy, hit_top, hit_bottom = wall_collide(by, bvy, h=court_h)

    bx, bvx, bvy, new_rally, hit_left = paddle_collide(
        bx, by, bvx, bvy, new_left_y, PADDLE_INSET, rally, going_left=True,
    )
    bx, bvx, bvy, new_rally, hit_right = paddle_collide(
        bx, by, bvx, bvy, new_right_y, court_w - PADDLE_INSET, new_rally, going_left=False,
    )

    scored_left, scored_right = score_detect(bx, w=court_w)
    scored_any = scored_left | scored_right

    new_score_left = score_left + scored_left.float()
    new_score_right = score_right + scored_right.float()

    # Auto-serve on score
    serve_bx, serve_by, serve_bvx, serve_bvy = serve_ball_from_rand(
        rand_angle, rand_dir, court_w, court_h,
    )
    bx = torch.where(scored_any, serve_bx, bx)
    by = torch.where(scored_any, serve_by, by)
    bvx = torch.where(scored_any, serve_bvx, bvx)
    bvy = torch.where(scored_any, serve_bvy, bvy)
    new_rally = torch.where(scored_any, torch.tensor(0.0), new_rally)

    game_over = (new_score_left >= MAX_SCORE) | (new_score_right >= MAX_SCORE)

    events = torch.stack([
        hit_left.float(), hit_right.float(),
        hit_top.float(), hit_bottom.float(),
        scored_left.float(), scored_right.float(),
    ])

    return (bx, by, bvx, bvy,
            new_left_y, new_right_y,
            new_score_left, new_score_right,
            new_rally,
            events, game_over.float())


def rule_based_policy(obs: Tensor, memory_y: Tensor, rand_val: Tensor,
                      reaction: float = 0.16, jitter: float = 22.0,
                      look_ahead: float = 0.14, threshold: float = 8.0,
                      court_h: _Scalar = COURT_H,
                      ) -> tuple[Tensor, Tensor]:
    """Core rule-based AI: obs → action + new_memory.
    Shared by PongPolicy (ONNX) and RuleBasedAgent (RL training)."""
    ball_y = obs[1] * court_h
    ball_vy = obs[3] * BALL_BASE_SPEED
    own_y = obs[4] * court_h
    new_memory = ai_track(ball_y, ball_vy, memory_y,
                          reaction, jitter, look_ahead,
                          rand_val, court_h=court_h)
    action = target_to_action(new_memory, own_y, threshold)
    return action, new_memory


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
