"""Reference motion library: G1 retargets (LAFAN1 or BONES-SEED) resampled to 50 Hz, with FK of tracked bodies."""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .kinematics import G1Kinematics
from .motion_io import CONTROL_DT, read_clip, resample
from .rotations import mat_to_6d, quat_to_mat, so3_log, yaw_mat

BIN_FRAMES = 50  # 1 s bins for return-based reset sampling
CACHE_BODIES_MAX_FRAMES = 5_000_000  # above this, tracked-body FK of the reference runs on demand (it alone is ~0.9 KB/frame)


class MotionLib:
    def __init__(self, names: list[str], p: torch.Tensor, quat: torch.Tensor, j: torch.Tensor, lens: list[int],
                 kin: G1Kinematics, device: str, cache_bodies: bool | None = None):  # fmt: skip
        """Clips at 50 Hz concatenated along dim 0: root position p (N,3), root quaternion wxyz (N,4), joints j (N,29)."""
        self.device = device
        self.kin = kin
        self.names = list(names)
        self.clip_len = torch.tensor(lens, dtype=torch.long, device=device)
        self.clip_start = torch.cumsum(self.clip_len, 0) - self.clip_len
        self.clip_end = self.clip_start + self.clip_len - 1  # last valid frame (inclusive)
        self.num_frames = int(self.clip_len.sum())
        self.clip_id = torch.repeat_interleave(torch.arange(len(lens), device=device), self.clip_len)

        p, j = p.to(device), j.to(device)
        R = quat_to_mat(quat.to(device))
        s, e = self.clip_start, self.clip_end
        # central differences inside each clip, one-sided at its first and last frame
        v, qd = _gradient(p, s, e, CONTROL_DT), _gradient(j, s, e, CONTROL_DT)
        w = torch.zeros_like(p)  # world angular velocity by central differences of rotations
        w[1:-1] = so3_log(R[2:] @ R[:-2].transpose(-1, -2)) / (2 * CONTROL_DT)
        w[s] = so3_log(R[s + 1] @ R[s].transpose(-1, -2)) / CONTROL_DT
        w[e] = so3_log(R[e] @ R[e - 1].transpose(-1, -2)) / CONTROL_DT
        self.root_p, self.root_R, self.root_v, self.root_w, self.q, self.qd = p, R, v, w, j, qd

        self.cache_bodies = self.num_frames <= CACHE_BODIES_MAX_FRAMES if cache_bodies is None else cache_bodies
        if self.cache_bodies:
            with torch.no_grad():
                out = [self._fk(torch.arange(a, min(a + 65536, self.num_frames), device=device)) for a in range(0, self.num_frames, 65536)]
                self.body_p, self.body_R, self.body_v, self.body_w = (torch.cat(x) for x in zip(*out))
        # per-frame static goal parts: q, qd, root velocities in the root frame
        Rt = self.root_R.transpose(-1, -2)
        self.root_v_local = (Rt @ self.root_v.unsqueeze(-1)).squeeze(-1)
        self.root_w_local = (Rt @ self.root_w.unsqueeze(-1)).squeeze(-1)
        self.root_gz = self.root_R[:, 2, 2]  # -(projected gravity z) of the reference root: g_b = R^T (0,0,-1)
        # 1 s bins
        nb = (self.clip_len + BIN_FRAMES - 1) // BIN_FRAMES
        first_bin = torch.cumsum(nb, 0) - nb
        self.bin_clip = torch.repeat_interleave(torch.arange(len(lens), device=device), nb)
        self.bin_start = self.clip_start[self.bin_clip] + (torch.arange(int(nb.sum()), device=device) - first_bin[self.bin_clip]) * BIN_FRAMES
        self.num_bins = int(nb.sum())
        frame = torch.arange(self.num_frames, device=device)
        self.frame_bin = first_bin[self.clip_id] + (frame - self.clip_start[self.clip_id]) // BIN_FRAMES

    def _fk(self, f: torch.Tensor):
        return self.kin.forward(self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f])

    def bodies(self, f: torch.Tensor):
        """Reference tracked-body positions, rotations, linear and angular velocities at frames f."""
        if self.cache_bodies:
            return self.body_p[f], self.body_R[f], self.body_v[f], self.body_w[f]
        with torch.no_grad():
            return self._fk(f)

    @staticmethod
    def from_files(files: list[str], kin: G1Kinematics, device: str) -> MotionLib:
        names, ps, qs, js = [], [], [], []
        for f in files:
            p, quat, j = resample(*read_clip(f))
            names.append(os.path.splitext(os.path.basename(f))[0])
            ps.append(p), qs.append(quat), js.append(j)
        return MotionLib(names, torch.cat(ps), torch.cat(qs), torch.cat(js), [len(p) for p in ps], kin, device)

    @staticmethod
    def from_packed(path: str, kin: G1Kinematics, device: str) -> MotionLib:
        """A library packed by tools/pack_motions.py (clips already at 50 Hz)."""
        z = np.load(path)
        t = lambda k: torch.from_numpy(z[k]).float()  # noqa: E731
        return MotionLib(z["names"].tolist(), t("p"), t("quat"), t("j"), z["lens"].tolist(), kin, device)

    @staticmethod
    def from_dir(path: str, kin: G1Kinematics, device: str) -> MotionLib:
        """A directory of CSVs (searched recursively), or a packed .npz library."""
        if path.endswith(".npz"):
            return MotionLib.from_packed(path, kin, device)
        files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        return MotionLib.from_files(files, kin, device)

    def future_frames(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Frames f+1..f+K clamped to the clip's last frame, shape (B, K)."""
        end = self.clip_end[self.clip_id[f]]
        fut = f.unsqueeze(-1) + torch.arange(1, K + 1, device=f.device)
        return torch.minimum(fut, end.unsqueeze(-1))

    def goal(self, f: torch.Tensor, K: int) -> torch.Tensor:
        """Goal features from the reference only: K future frames relative to the current reference frame."""
        fut = self.future_frames(f, K)  # (B,K)
        H = yaw_mat(self.root_R[f])  # (B,3,3)
        Ht = H.transpose(-1, -2).unsqueeze(1)
        dp = (Ht @ (self.root_p[fut] - self.root_p[f].unsqueeze(1)).unsqueeze(-1)).squeeze(-1)
        rot = mat_to_6d(Ht @ self.root_R[fut])
        feats = torch.cat(
            (dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1
        )  # (B,K,73)
        return feats.reshape(f.shape[0], -1)

    def ref_state(self, f: torch.Tensor):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]


def _gradient(x: torch.Tensor, s: torch.Tensor, e: torch.Tensor, dt: float) -> torch.Tensor:
    """Per-clip finite differences of x (N,...) for clips spanning frames s..e (inclusive)."""
    g = torch.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[s] = (x[s + 1] - x[s]) / dt
    g[e] = (x[e] - x[e - 1]) / dt
    return g
