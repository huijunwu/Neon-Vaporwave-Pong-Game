"""
Export Pong ONNX models.

Usage:
    cd src/python
    pip install -e .
    python scripts/export_onnx.py

Output:  dist/assets/onnx/pong_step.onnx, pong_policy.onnx
"""

import os
import torch
from pong.onnx_modules import PongStep, PongPolicy

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "dist", "assets", "onnx")
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


def export_step():
    _export(
        PongStep(),
        (
            torch.tensor(400.0),    # ball_x
            torch.tensor(300.0),    # ball_y
            torch.tensor(-500.0),   # ball_vx
            torch.tensor(200.0),    # ball_vy
            torch.tensor(300.0),    # paddle_left_y
            torch.tensor(300.0),    # paddle_right_y
            torch.tensor(0.0),      # action_left (float32)
            torch.tensor(1.0),      # action_right (float32)
            torch.tensor(5.0),      # rally
            torch.tensor(800.0),    # W
            torch.tensor(600.0),    # H
        ),
        "pong_step.onnx",
        ["ball_x", "ball_y", "ball_vx", "ball_vy",
         "paddle_left_y", "paddle_right_y",
         "action_left", "action_right",
         "rally", "W", "H"],
        ["new_ball_x", "new_ball_y", "new_ball_vx", "new_ball_vy",
         "new_paddle_left_y", "new_paddle_right_y",
         "new_rally", "events"],
    )


def export_policy():
    _export(
        PongPolicy(),
        (
            torch.tensor([0.5, 0.5, -0.5, 0.2, 0.5, 0.5]),  # obs[6]
            torch.tensor(300.0),                                # memory_y
            torch.tensor(0.3),                                  # rand_val
            torch.tensor(600.0),                                # H
        ),
        "pong_policy.onnx",
        ["obs", "memory_y", "rand_val", "H"],
        ["action", "new_memory_y"],
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Exporting to {os.path.abspath(OUTPUT_DIR)}/")
    export_step()
    export_policy()
    print("Done.")


if __name__ == "__main__":
    main()
