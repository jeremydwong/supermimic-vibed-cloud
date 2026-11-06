"""Reading G1 retarget CSVs (LAFAN1 or BONES-SEED) and resampling them to the control rate; no Isaac Lab needed."""

from __future__ import annotations

import numpy as np
import torch

from .rotations import mat_to_quat, quat_slerp

CONTROL_DT = 0.02
BONES_FPS = 120.0
LAFAN1_FPS = 30.0


def euler_xyz_deg_to_quat(e: torch.Tensor) -> torch.Tensor:
    """Extrinsic x-y-z Euler angles in degrees (R = Rz Ry Rx) -> quaternion wxyz."""
    a, b, c = torch.deg2rad(e).unbind(-1)
    ca, sa, cb, sb, cc, sc = a.cos(), a.sin(), b.cos(), b.sin(), c.cos(), c.sin()
    R = torch.stack(
        (
            cc * cb, cc * sb * sa - sc * ca, cc * sb * ca + sc * sa,
            sc * cb, sc * sb * sa + cc * ca, sc * sb * ca - cc * sa,
            -sb, cb * sa, cb * ca,
        ),
        dim=-1,
    ).reshape(e.shape[:-1] + (3, 3))  # fmt: skip
    return mat_to_quat(R)


def read_clip(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """One G1 retarget CSV -> root position (T,3) m, root quaternion wxyz (T,4), joints (T,29) rad, source fps.

    LAFAN1 (no header, 30 fps): x y z [m], qx qy qz qw, 29 joints [rad].
    BONES-SEED (header row, 120 fps): Frame, root translate [cm], root rotate (extrinsic xyz) [deg], 29 joints [deg].
    """
    with open(path) as fh:
        bones = fh.readline().startswith("Frame,")
    raw = torch.tensor(np.loadtxt(path, delimiter=",", skiprows=int(bones), ndmin=2), dtype=torch.float32)
    if bones:
        return raw[:, 1:4] / 100.0, euler_xyz_deg_to_quat(raw[:, 4:7]), torch.deg2rad(raw[:, 7:36]), BONES_FPS
    q_xyzw = raw[:, 3:7]
    return raw[:, :3], torch.cat((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1), raw[:, 7:36], LAFAN1_FPS


def resample(p: torch.Tensor, quat: torch.Tensor, j: torch.Tensor, src_fps: float):
    """Linear (slerp for the root rotation) resampling to the 50 Hz control rate."""
    T_src = p.shape[0]
    dur = (T_src - 1) / src_fps
    n = int(np.floor(dur / CONTROL_DT)) + 1
    t = torch.arange(n, dtype=torch.float32) * CONTROL_DT * src_fps
    i0 = torch.clamp(t.floor().long(), max=T_src - 2)
    a = (t - i0.float()).clamp(0, 1)
    p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
    j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
    return p, quat_slerp(quat[i0], quat[i0 + 1], a), j
