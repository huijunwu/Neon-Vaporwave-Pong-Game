"""
ONNX export wrappers — thin nn.Module shells over physics.py functions.

Only 2 models:
  PongStep    → pong_step.onnx     (env physics, symmetric)
  PongPolicy  → pong_policy.onnx   (agent, currently rule-based)
"""

import torch
import torch.nn as nn

from pong.physics import (
    COURT_W, COURT_H, PADDLE_INSET, BALL_BASE_SPEED, PADDLE_H,
    ball_move, wall_collide, paddle_collide, score_detect, apply_action,
    ai_track, target_to_action,
)


class PongStep(nn.Module):
    """env.step() as ONNX: state + 2 discrete actions → new state + events.

    Input:  ball_x, ball_y, ball_vx, ball_vy,
            paddle_left_y, paddle_right_y,
            action_left, action_right,        ← float32: 0=NOOP, 1=DOWN(Y+), 2=UP(Y-)
            rally, W, H

    Output: new_ball_x, new_ball_y, new_ball_vx, new_ball_vy,
            new_paddle_left_y, new_paddle_right_y,
            new_rally,
            events[6]: [hit_left, hit_right, hit_top, hit_bottom, scored_left, scored_right]
    """

    def forward(self, ball_x, ball_y, ball_vx, ball_vy,
                paddle_left_y, paddle_right_y,
                action_left, action_right,
                rally, W, H):
        new_left_y = apply_action(paddle_left_y, action_left, court_h=H)
        new_right_y = apply_action(paddle_right_y, action_right, court_h=H)

        bx, by = ball_move(ball_x, ball_y, ball_vx, ball_vy)
        bvx = ball_vx
        bvy = ball_vy

        by, bvy, hit_top, hit_bottom = wall_collide(by, bvy, h=H)

        bx, bvx, bvy, new_rally, hit_left = paddle_collide(
            bx, by, bvx, bvy, new_left_y, PADDLE_INSET, rally, going_left=True,
        )
        bx, bvx, bvy, new_rally, hit_right = paddle_collide(
            bx, by, bvx, bvy, new_right_y, W - PADDLE_INSET, new_rally, going_left=False,
        )

        scored_left, scored_right = score_detect(bx, w=W)

        events = torch.stack([
            hit_left.float(), hit_right.float(),
            hit_top.float(), hit_bottom.float(),
            scored_left.float(), scored_right.float(),
        ])

        return (bx, by, bvx, bvy,
                new_left_y, new_right_y,
                new_rally, events)


class PongPolicy(nn.Module):
    """Rule-based agent as ONNX. Same interface as future neural network policy.

    Input:  obs[6] = [ball_x/W, ball_y/H, ball_vx/S, ball_vy/S, own_y/H, opp_y/H]
            memory_y, rand_val, H

    Output: action (float32: 0=NOOP, 1=DOWN(Y+), 2=UP(Y-)), new_memory_y
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
        ball_y = obs[1] * H
        ball_vy = obs[3] * BALL_BASE_SPEED
        own_y = obs[4] * H

        new_memory = ai_track(
            ball_y, ball_vy, memory_y,
            self.reaction, self.jitter, self.look_ahead,
            rand_val, court_h=H,
        )

        action = target_to_action(new_memory, own_y, self.threshold)

        return action, new_memory
