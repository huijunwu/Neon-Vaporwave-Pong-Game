"""
Pong game logic as PyTorch modules — 1:1 mapping to JS functions.

JS function              → PyTorch Module         → ONNX file
─────────────────────────────────────────────────────────────
clamp(v, a, b)           → Clamp                  → pong_clamp.onnx
lerp(a, b, t)            → Lerp                   → pong_lerp.onnx
shock(a)                 → Shock                  → pong_shock.onnx
aiUpdate(dt)             → AIUpdate               → pong_ai.onnx
ball physics + collision → BallPhysics            → pong_physics.onnx
updateCutscene(dt)       → CutsceneUpdate         → pong_cutscene.onnx
updateParticles(dt)      → ParticleUpdate         → pong_particles.onnx
"""

import torch
import torch.nn as nn

# ── Game constants (from JS source, never change during gameplay) ──
PADDLE_W = 14.0
PADDLE_INSET = 26.0
BALL_R = 10.0
BALL_BASE_SPEED = 560.0


def level_from_progress(score_L, score_R):
    """JS: levelFromProgress() — AI difficulty 0‑4 based on scores."""
    total = score_L + score_R
    max_score = torch.max(score_L, score_R)
    lvl = torch.tensor(0.0)
    lvl = torch.where(total >= 3, torch.tensor(1.0), lvl)
    lvl = torch.where(total >= 7, torch.tensor(2.0), lvl)
    lvl = torch.where(max_score >= 6, torch.tensor(3.0), lvl)
    lvl = torch.where(max_score >= 9, torch.tensor(4.0), lvl)
    return lvl


def _lookup(vals, lvl):
    """ONNX-safe lookup: nested torch.where instead of dynamic indexing.
    vals: list of 5 floats, lvl: scalar tensor 0-4."""
    result = torch.tensor(vals[0])
    result = torch.where(lvl >= 1, torch.tensor(vals[1]), result)
    result = torch.where(lvl >= 2, torch.tensor(vals[2]), result)
    result = torch.where(lvl >= 3, torch.tensor(vals[3]), result)
    result = torch.where(lvl >= 4, torch.tensor(vals[4]), result)
    return result


# ═══════════════════════════════════════════════════════════════════
#  1. Clamp  ←  JS clamp(v, a, b)
# ═══════════════════════════════════════════════════════════════════

class Clamp(nn.Module):
    """JS: Math.max(a, Math.min(b, v))"""

    def forward(self, v, a, b):
        return torch.max(a, torch.min(v, b))


# ═══════════════════════════════════════════════════════════════════
#  2. Lerp  ←  JS lerp(a, b, t)
# ═══════════════════════════════════════════════════════════════════

class Lerp(nn.Module):
    """JS: a + (b - a) * t"""

    def forward(self, a, b, t):
        return a + (b - a) * t


# ═══════════════════════════════════════════════════════════════════
#  3. Shock  ←  JS shock(a)
# ═══════════════════════════════════════════════════════════════════

class Shock(nn.Module):
    """JS: state.shake = Math.min(10, state.shake + a)"""

    def forward(self, shake, a):
        return torch.min(torch.tensor(10.0), shake + a)


# ═══════════════════════════════════════════════════════════════════
#  4. AIUpdate  ←  JS aiUpdate(dt)
# ═══════════════════════════════════════════════════════════════════

class AIUpdate(nn.Module):
    """AI paddle movement: predict target, smooth tracking, clamp speed."""

    def forward(self, ball_y, ball_vy, ball_vx,
                ai_y, ai_memoryY,
                score_L, score_R,
                dt, H, rand_val):
        """
        rand_val: float in [-1, 1]  (JS passes Math.random()*2-1)
        Returns: (new_ai_y, new_ai_memoryY, ai_vy, ai_h)
        """
        lvl = level_from_progress(score_L, score_R)

        reaction   = _lookup([0.24, 0.20, 0.16, 0.13, 0.12], lvl)
        max_speed  = _lookup([520., 600., 700., 820., 920.], lvl)
        jitter     = _lookup([42.,  32.,  22.,  14.,  10.],  lvl)
        look_ahead = _lookup([0.10, 0.12, 0.14, 0.16, 0.17], lvl)

        # JS: ai.h = clamp(118 - lvl*6, 88, 118)
        ai_h = torch.clamp(118.0 - lvl * 6.0, 88.0, 118.0)

        # Target: track ball when approaching AI side, else center
        coming = ball_vx > 0
        target = torch.where(
            coming,
            ball_y + ball_vy * look_ahead + rand_val * jitter,
            H / 2.0,
        )
        target = torch.clamp(target, ai_h / 2.0 + 8.0, H - ai_h / 2.0 - 8.0)

        # JS: lerp(ai.memoryY, target, 1 - pow(0.0006, dt/reaction))
        t = 1.0 - torch.pow(torch.tensor(0.0006), dt / reaction)
        new_memoryY = ai_memoryY + (target - ai_memoryY) * t

        # Velocity & position
        dy = new_memoryY - ai_y
        ai_vy = torch.clamp(dy / reaction, -max_speed, max_speed)
        new_ai_y = ai_y + ai_vy * dt

        return new_ai_y, new_memoryY, ai_vy, ai_h


# ═══════════════════════════════════════════════════════════════════
#  2. BallPhysics  ←  JS update() physics block
#     Includes: paddle smoothing, ball movement, wall/paddle
#     collision, score detection
# ═══════════════════════════════════════════════════════════════════

class BallPhysics(nn.Module):
    """Ball movement + wall collision + paddle collision + scoring."""

    def forward(self, ball_x, ball_y, ball_vx, ball_vy,
                paddle_y, paddle_target_y, paddle_h,
                ai_y, ai_h,
                score_L, score_R, rally,
                dt, W, H):
        """
        Returns:
            new_ball_x, new_ball_y, new_ball_vx, new_ball_vy,
            new_paddle_y, new_rally,
            events[6]: [hit_player, hit_ai, wall_top, wall_bottom, scored_L, scored_R]
        """
        # ── Player paddle smoothing ──
        # JS: paddle.y = lerp(paddle.y, paddle.targetY, 1-pow(0.0009,dt))
        ease = 1.0 - torch.pow(torch.tensor(0.0009), dt)
        new_paddle_y = paddle_y + (paddle_target_y - paddle_y) * ease

        # ── Ball movement ──
        bx = ball_x + ball_vx * dt
        by = ball_y + ball_vy * dt
        bvx = ball_vx
        bvy = ball_vy

        # ── Wall collision (top) ──
        hit_top = by - BALL_R <= 0
        by  = torch.where(hit_top, torch.tensor(BALL_R), by)
        bvy = torch.where(hit_top, -bvy, bvy)

        # ── Wall collision (bottom) ──
        hit_bottom = by + BALL_R >= H
        by  = torch.where(hit_bottom, H - BALL_R, by)
        bvy = torch.where(hit_bottom, -bvy, bvy)

        # ── Difficulty level (for speed ramp) ──
        lvl = level_from_progress(score_L, score_R)

        # ── Player paddle collision ──
        # JS: if (ball.vx<0 && ball.x-ball.r <= px+paddle.w/2)
        #       && ball.y in [top, bot]
        px = PADDLE_INSET
        hit_player = (bvx < 0) \
            & ((bx - BALL_R) <= (px + PADDLE_W / 2.0)) \
            & (by >= new_paddle_y - paddle_h / 2.0) \
            & (by <= new_paddle_y + paddle_h / 2.0)

        p_n     = (by - new_paddle_y) / (paddle_h / 2.0)
        p_angle = p_n * 0.95
        p_up    = _lookup([1.02, 1.03, 1.035, 1.04, 1.045], lvl)
        p_rally = rally + 1.0
        p_speed = torch.clamp(
            BALL_BASE_SPEED * (p_up + p_rally * 0.0016), 560.0, 980.0
        )
        p_vx = torch.abs(bvx)
        p_vy = p_angle * p_speed
        p_mag = torch.sqrt(p_vx * p_vx + p_vy * p_vy).clamp(min=1.0)
        p_vx = (p_vx / p_mag) * p_speed
        p_vy = (p_vy / p_mag) * p_speed

        bx  = torch.where(hit_player, torch.tensor(px + PADDLE_W / 2.0 + BALL_R), bx)
        bvx = torch.where(hit_player, p_vx, bvx)
        bvy = torch.where(hit_player, p_vy, bvy)
        new_rally = torch.where(hit_player, p_rally, rally)

        # ── AI paddle collision ──
        # JS: if (ball.vx>0 && ball.x+ball.r >= ax-ai.w/2)
        ax = W - PADDLE_INSET
        hit_ai = (bvx > 0) \
            & ((bx + BALL_R) >= (ax - PADDLE_W / 2.0)) \
            & (by >= ai_y - ai_h / 2.0) \
            & (by <= ai_y + ai_h / 2.0)

        a_n     = (by - ai_y) / (ai_h / 2.0)
        a_angle = a_n * 0.9
        a_up    = _lookup([1.015, 1.025, 1.03, 1.035, 1.04], lvl)
        a_rally = new_rally + 1.0
        a_speed = torch.clamp(
            BALL_BASE_SPEED * (a_up + a_rally * 0.0012), 560.0, 960.0
        )
        a_vx = -torch.abs(bvx)
        a_vy = a_angle * a_speed
        a_mag = torch.sqrt(a_vx * a_vx + a_vy * a_vy).clamp(min=1.0)
        a_vx = (a_vx / a_mag) * a_speed
        a_vy = (a_vy / a_mag) * a_speed

        bx  = torch.where(hit_ai, ax - PADDLE_W / 2.0 - BALL_R, bx)
        bvx = torch.where(hit_ai, a_vx, bvx)
        bvy = torch.where(hit_ai, a_vy, bvy)
        new_rally = torch.where(hit_ai, a_rally, new_rally)

        # ── Score detection ──
        scored_L = (bx > W + 60.0)    # ball exits right → player scores
        scored_R = (bx < -60.0)        # ball exits left  → AI scores

        events = torch.stack([
            hit_player.float(),
            hit_ai.float(),
            hit_top.float(),
            hit_bottom.float(),
            scored_L.float(),
            scored_R.float(),
        ])

        return bx, by, bvx, bvy, new_paddle_y, new_rally, events


# ═══════════════════════════════════════════════════════════════════
#  3. CutsceneUpdate  ←  JS updateCutscene(dt)
# ═══════════════════════════════════════════════════════════════════

class CutsceneUpdate(nn.Module):
    """Match-point slowmo transition."""

    def forward(self, cut, cutT, slowmo, dt):
        """
        cut: 0.0 or 1.0 (bool as float)
        Returns: (new_cut, new_cutT, new_slowmo, cut_ended)
        """
        is_cut = cut > 0.5

        new_cutT = torch.where(is_cut, cutT + dt, cutT)

        # After 0.9s: ramp slowmo back toward 1.0
        # JS: lerp(slowmo, 1, 1 - pow(0.0009, dt))
        t = 1.0 - torch.pow(torch.tensor(0.0009), dt)
        lerped = slowmo + (1.0 - slowmo) * t
        new_slowmo = torch.where(is_cut & (new_cutT > 0.9), lerped, slowmo)

        # After 1.35s: end cutscene
        should_end = is_cut & (new_cutT > 1.35)
        new_cut    = torch.where(should_end, torch.tensor(0.0), cut)
        new_slowmo = torch.where(should_end, torch.tensor(1.0), new_slowmo)

        cut_ended = should_end.float()

        return new_cut, new_cutT, new_slowmo, cut_ended


# ═══════════════════════════════════════════════════════════════════
#  8. ParticleUpdate  ←  JS updateParticles(dt)
#     Fixed-size tensors: particles [MAX_PARTICLES, 7],
#     sparks [MAX_SPARKS, 7].  Column order:
#     [x, y, vx, vy, life, t, alive]
#     JS manages push/splice via alive mask; ONNX does the math.
# ═══════════════════════════════════════════════════════════════════

MAX_PARTICLES = 70
MAX_SPARKS = 40


class ParticleUpdate(nn.Module):
    """JS: updateParticles(dt) — tick positions, decay velocities, expire."""

    def forward(self, particles, sparks, dt):
        """
        particles: float32[MAX_PARTICLES, 7]
        sparks:    float32[MAX_SPARKS, 7]
        Returns:   (new_particles, new_sparks) same shapes
        """
        # ── Particles (decay factor 0.14^dt) ──
        p_x, p_y, p_vx, p_vy, p_life, p_t, p_alive = particles.unbind(dim=1)

        p_decay = torch.pow(torch.tensor(0.14), dt)
        new_p_t = p_t + dt
        new_p_x = p_x + p_vx * dt
        new_p_y = p_y + p_vy * dt
        new_p_vx = p_vx * p_decay
        new_p_vy = p_vy * p_decay
        new_p_alive = p_alive * (new_p_t < p_life).float()

        new_particles = torch.stack(
            [new_p_x, new_p_y, new_p_vx, new_p_vy, p_life, new_p_t, new_p_alive],
            dim=1,
        )

        # ── Sparks (decay factor 0.06^dt) ──
        s_x, s_y, s_vx, s_vy, s_life, s_t, s_alive = sparks.unbind(dim=1)

        s_decay = torch.pow(torch.tensor(0.06), dt)
        new_s_t = s_t + dt
        new_s_x = s_x + s_vx * dt
        new_s_y = s_y + s_vy * dt
        new_s_vx = s_vx * s_decay
        new_s_vy = s_vy * s_decay
        new_s_alive = s_alive * (new_s_t < s_life).float()

        new_sparks = torch.stack(
            [new_s_x, new_s_y, new_s_vx, new_s_vy, s_life, new_s_t, new_s_alive],
            dim=1,
        )

        return new_particles, new_sparks
