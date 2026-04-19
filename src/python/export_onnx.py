"""
Export all Pong game logic modules to ONNX.

Usage:
    cd src/python
    pip install -r requirements.txt
    python export_onnx.py

Output:  ../../dist/onnx/*.onnx
"""

import os
import torch
from models import (
    Clamp, Lerp, Shock,
    AIUpdate, BallPhysics, CutsceneUpdate,
    ParticleUpdate, MAX_PARTICLES, MAX_SPARKS,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "dist", "onnx")
OPSET = 17


def _export(model, args, filename, input_names, output_names):
    model.eval()
    torch.onnx.export(
        model, args,
        os.path.join(OUTPUT_DIR, filename),
        input_names=input_names,
        output_names=output_names,
        opset_version=OPSET,
    )
    print(f"  {filename}")


def export_clamp():
    _export(
        Clamp(),
        (torch.tensor(5.0), torch.tensor(0.0), torch.tensor(10.0)),
        "pong_clamp.onnx",
        ["v", "a", "b"],
        ["result"],
    )


def export_lerp():
    _export(
        Lerp(),
        (torch.tensor(0.0), torch.tensor(100.0), torch.tensor(0.5)),
        "pong_lerp.onnx",
        ["a", "b", "t"],
        ["result"],
    )


def export_shock():
    _export(
        Shock(),
        (torch.tensor(3.0), torch.tensor(4.0)),
        "pong_shock.onnx",
        ["shake", "amount"],
        ["new_shake"],
    )


def export_ai():
    _export(
        AIUpdate(),
        (
            torch.tensor(300.0),    # ball_y
            torch.tensor(100.0),    # ball_vy
            torch.tensor(-400.0),   # ball_vx
            torch.tensor(300.0),    # ai_y
            torch.tensor(300.0),    # ai_memoryY
            torch.tensor(2.0),      # score_L
            torch.tensor(1.0),      # score_R
            torch.tensor(0.016),    # dt
            torch.tensor(600.0),    # H
            torch.tensor(0.3),      # rand_val
        ),
        "pong_ai.onnx",
        ["ball_y", "ball_vy", "ball_vx",
         "ai_y", "ai_memoryY",
         "score_L", "score_R",
         "dt", "H", "rand_val"],
        ["new_ai_y", "new_ai_memoryY", "ai_vy", "ai_h"],
    )


def export_physics():
    _export(
        BallPhysics(),
        (
            torch.tensor(400.0),    # ball_x
            torch.tensor(300.0),    # ball_y
            torch.tensor(-500.0),   # ball_vx
            torch.tensor(200.0),    # ball_vy
            torch.tensor(300.0),    # paddle_y
            torch.tensor(280.0),    # paddle_target_y
            torch.tensor(110.0),    # paddle_h
            torch.tensor(300.0),    # ai_y
            torch.tensor(110.0),    # ai_h
            torch.tensor(3.0),      # score_L
            torch.tensor(2.0),      # score_R
            torch.tensor(5.0),      # rally
            torch.tensor(0.016),    # dt
            torch.tensor(800.0),    # W
            torch.tensor(600.0),    # H
        ),
        "pong_physics.onnx",
        ["ball_x", "ball_y", "ball_vx", "ball_vy",
         "paddle_y", "paddle_target_y", "paddle_h",
         "ai_y", "ai_h",
         "score_L", "score_R", "rally",
         "dt", "W", "H"],
        ["new_ball_x", "new_ball_y", "new_ball_vx", "new_ball_vy",
         "new_paddle_y", "new_rally", "events"],
    )


def export_cutscene():
    _export(
        CutsceneUpdate(),
        (
            torch.tensor(1.0),     # cut
            torch.tensor(0.5),     # cutT
            torch.tensor(0.45),    # slowmo
            torch.tensor(0.016),   # dt
        ),
        "pong_cutscene.onnx",
        ["cut", "cutT", "slowmo", "dt"],
        ["new_cut", "new_cutT", "new_slowmo", "cut_ended"],
    )


def export_particles():
    _export(
        ParticleUpdate(),
        (
            torch.zeros(MAX_PARTICLES, 7),  # particles
            torch.zeros(MAX_SPARKS, 7),     # sparks
            torch.tensor(0.016),            # dt
        ),
        "pong_particles.onnx",
        ["particles", "sparks", "dt"],
        ["new_particles", "new_sparks"],
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Exporting to {os.path.abspath(OUTPUT_DIR)}/")
    export_clamp()
    export_lerp()
    export_shock()
    export_ai()
    export_physics()
    export_cutscene()
    export_particles()
    print("Done.")


if __name__ == "__main__":
    main()
