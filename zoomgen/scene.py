from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from .config import GeneratorConfig


REQUIRED = {
    "x", "y", "z", "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3", "opacity",
    "f_dc_0", "f_dc_1", "f_dc_2",
}


class MissingRestFieldsError(ValueError):
    """Raised when a Gaussian PLY only contains degree-0 SH colors."""


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid for potentially extreme logits.
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out


@dataclass
class SceneAnalysis:
    original_gaussian_count: int
    effective_gaussian_count: int
    filtered_gaussian_count: int
    nonfinite_count: int
    low_opacity_count: int
    position_outlier_count: int
    center: np.ndarray
    radius: float
    aabb_min: np.ndarray
    aabb_max: np.ndarray
    position_filter_min: np.ndarray
    position_filter_max: np.ndarray
    opacity_storage: str
    sh_degree: int
    scale_cap: np.ndarray

    def as_dict(self) -> dict:
        return {
            "original_gaussian_count": self.original_gaussian_count,
            "effective_gaussian_count": self.effective_gaussian_count,
            "filtered_gaussian_count": self.filtered_gaussian_count,
            "nonfinite_count": self.nonfinite_count,
            "low_opacity_count": self.low_opacity_count,
            "position_outlier_count": self.position_outlier_count,
            "center": self.center.tolist(),
            "radius": self.radius,
            "aabb": {"min": self.aabb_min.tolist(), "max": self.aabb_max.tolist()},
            "position_filter_bounds": {
                "min": self.position_filter_min.tolist(), "max": self.position_filter_max.tolist()
            },
            "opacity_storage": self.opacity_storage,
            "sh_degree": self.sh_degree,
            "scale_cap": self.scale_cap.tolist(),
        }


@dataclass
class GaussianTensors:
    xyz: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacity: torch.Tensor
    shs: torch.Tensor
    sh_degree: int


class GaussianScene:
    def __init__(self, path: Path, cfg: GeneratorConfig):
        self.path = path
        self.cfg = cfg
        root = cfg.scene.root_transform
        self.root_rotation = Rotation.from_euler(
            root.rotation_order, root.rotation_euler_deg, degrees=True
        ).as_matrix()
        self.root_translation = np.asarray(root.translation, dtype=np.float64)
        self.root_scale = float(root.scale)
        try:
            self.ply = PlyData.read(str(path), mmap="r")
        except Exception as exc:
            raise ValueError(f"failed to read PLY {path}: {exc}") from exc
        if "vertex" not in self.ply:
            raise ValueError("PLY has no vertex element; expected a 3DGS vertex table")
        self.data = self.ply["vertex"].data
        self.names = tuple(self.data.dtype.names or ())
        missing = sorted(REQUIRED - set(self.names))
        if missing:
            raise ValueError(f"not a valid 3D Gaussian Splatting PLY (missing fields: {missing})")
        rest = sorted((x for x in self.names if x.startswith("f_rest_")), key=lambda x: int(x[7:]))
        try:
            self.rest_names, self.available_sh_degree = self._sh_layout(rest)
        except MissingRestFieldsError:
            # Some exporters store valid Gaussian geometry and degree-0 (DC) color
            # without allocating unused higher-order spherical harmonics fields.
            self.rest_names = []
            self.available_sh_degree = 0
            warnings.warn(
                f"PLY {path} has no f_rest_* fields; falling back to DC-only "
                "spherical harmonics (SH degree 0)",
                RuntimeWarning,
                stacklevel=2,
            )
        self.all_required_names = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
                                   "scale_0", "scale_1", "scale_2", "opacity",
                                   "f_dc_0", "f_dc_1", "f_dc_2", *self.rest_names]
        self.analysis = self._analyze()

    @staticmethod
    def _sh_layout(rest: list[str]) -> tuple[list[str], int]:
        if not rest:
            raise MissingRestFieldsError("missing f_rest_* fields")
        expected = list(range(len(rest)))
        got = [int(x[7:]) for x in rest]
        if got != expected or len(rest) % 3:
            raise ValueError("f_rest_* fields must be contiguous and divisible into 3 channels")
        coeff = 1 + len(rest) // 3
        root = int(round(math.sqrt(coeff)))
        if root * root != coeff:
            raise ValueError(f"invalid spherical harmonics field count: {len(rest)}")
        return rest, root - 1

    def _matrix(self, sl: slice, names: list[str]) -> np.ndarray:
        matrix = np.column_stack(
            [np.asarray(self.data[name][sl], dtype=np.float32) for name in names]
        )
        if all(name in names for name in ("x", "y", "z")):
            indices = [names.index(name) for name in ("x", "y", "z")]
            xyz = matrix[:, indices].astype(np.float64, copy=False)
            transformed = (
                self.root_scale * (self.root_rotation @ xyz.T).T + self.root_translation
            )
            matrix[:, indices] = transformed.astype(np.float32)
        return matrix

    def _opacity(self, raw: np.ndarray, storage: str) -> np.ndarray:
        return sigmoid(raw) if storage == "logit" else np.clip(raw, 0.0, 1.0)

    def _analyze(self) -> SceneAnalysis:
        n = len(self.data)
        acfg = self.cfg.scene_analysis
        stride = max(1, math.ceil(n / acfg.analysis_max_samples))
        sample_sl = slice(0, n, stride)
        pos = self._matrix(sample_sl, ["x", "y", "z"])
        log_scale = self._matrix(sample_sl, ["scale_0", "scale_1", "scale_2"])
        raw_opacity = np.asarray(self.data["opacity"][sample_sl], dtype=np.float32)
        core_finite = np.isfinite(pos).all(1) & np.isfinite(log_scale).all(1) & np.isfinite(raw_opacity)
        finite_op = raw_opacity[core_finite]
        if finite_op.size == 0:
            raise ValueError("PLY contains no finite Gaussian core fields")
        opacity_storage = "probability" if finite_op.min() >= 0 and finite_op.max() <= 1 else "logit"
        alpha = self._opacity(raw_opacity, opacity_storage)
        visible = core_finite & (alpha >= acfg.opacity_threshold)
        if not np.any(visible):
            raise ValueError("no Gaussians remain after opacity/non-finite filtering")
        p = pos[visible]
        lo_q, hi_q = acfg.position_quantiles
        pos_lo = np.quantile(p, lo_q, axis=0)
        pos_hi = np.quantile(p, hi_q, axis=0)
        robust_sample = visible & np.all(pos >= pos_lo, axis=1) & np.all(pos <= pos_hi, axis=1)
        # Per-axis quantile intersections can be empty for tiny, diagonal samples.
        # Use all already finite/opaque samples for the estimate in that edge case;
        # the exact filtering pass still applies the robust position bounds.
        if int(robust_sample.sum()) < min(4, int(visible.sum())):
            robust_sample = visible
            pos_lo = np.min(pos[visible], axis=0)
            pos_hi = np.max(pos[visible], axis=0)
        p = pos[robust_sample]
        scales = self.root_scale * np.exp(
            np.clip(log_scale[robust_sample], -30.0, 30.0)
        )
        scale_cap = np.quantile(scales, acfg.max_scale_quantile, axis=0)
        clipped_scales = np.minimum(scales, scale_cap)
        center = np.median(p, axis=0)
        aabb_min = np.quantile(p - acfg.scale_extent_sigma * clipped_scales, lo_q, axis=0)
        aabb_max = np.quantile(p + acfg.scale_extent_sigma * clipped_scales, hi_q, axis=0)
        corners = np.array(np.meshgrid(*zip(aabb_min, aabb_max))).T.reshape(-1, 3)
        radius = float(np.max(np.linalg.norm(corners - center, axis=1)))
        radius = max(radius, float(np.finfo(np.float32).eps))

        effective = nonfinite = low_opacity = outliers = 0
        for start in tqdm(range(0, n, acfg.io_chunk_size), desc="Analyzing PLY", unit="chunk"):
            stop = min(n, start + acfg.io_chunk_size)
            core = self._matrix(slice(start, stop), self.all_required_names)
            finite = np.isfinite(core).all(1)
            op = self._opacity(core[:, 10], opacity_storage)
            opaque = op >= acfg.opacity_threshold
            inside = np.all(core[:, :3] >= pos_lo, axis=1) & np.all(core[:, :3] <= pos_hi, axis=1)
            valid = (
                finite & opaque & inside
                if acfg.filter_gaussians
                else finite
            )
            effective += int(valid.sum())
            nonfinite += int((~finite).sum())
            low_opacity += int((finite & ~opaque).sum())
            outliers += int((finite & opaque & ~inside).sum())
        if effective == 0:
            raise ValueError("no effective Gaussians remain after robust filtering")
        return SceneAnalysis(n, effective, n - effective, nonfinite, low_opacity, outliers,
                             center, radius, aabb_min, aabb_max, pos_lo, pos_hi,
                             opacity_storage, self.available_sh_degree, scale_cap)

    def _valid_mask(self, block: np.ndarray) -> np.ndarray:
        a = self.analysis
        finite = np.isfinite(block).all(1)
        if not self.cfg.scene_analysis.filter_gaussians:
            return finite
        opaque = self._opacity(block[:, 10], a.opacity_storage) >= self.cfg.scene_analysis.opacity_threshold
        inside = np.all(block[:, :3] >= a.position_filter_min, axis=1) & np.all(
            block[:, :3] <= a.position_filter_max, axis=1
        )
        return finite & opaque & inside

    def effective_position_sample(
        self, max_points: int = 250_000, opacity_threshold: float | None = None
    ) -> np.ndarray:
        """Return a deterministic transformed-world sample for camera occupancy queries."""
        stride = max(1, math.ceil(len(self.data) / max_points))
        names = ["x", "y", "z", "opacity"]
        block = self._matrix(slice(0, len(self.data), stride), names)
        threshold = (
            self.cfg.scene_analysis.opacity_threshold
            if opacity_threshold is None
            else opacity_threshold
        )
        finite = np.isfinite(block).all(axis=1)
        opaque = self._opacity(block[:, 3], self.analysis.opacity_storage) >= threshold
        inside = np.all(
            block[:, :3] >= self.analysis.position_filter_min, axis=1
        ) & np.all(block[:, :3] <= self.analysis.position_filter_max, axis=1)
        return np.ascontiguousarray(block[finite & opaque & inside, :3])


    def load_tensors(self) -> GaussianTensors:
        device = torch.device(self.cfg.render.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("render.device is CUDA but torch.cuda.is_available() is false")
        n = self.analysis.effective_gaussian_count
        coeff = (self.available_sh_degree + 1) ** 2
        xyz = torch.empty((n, 3), device=device, dtype=torch.float32)
        scales = torch.empty((n, 3), device=device, dtype=torch.float32)
        rotations = torch.empty((n, 4), device=device, dtype=torch.float32)
        opacity = torch.empty((n, 1), device=device, dtype=torch.float32)
        shs = torch.empty((n, coeff, 3), device=device, dtype=torch.float32)
        acfg = self.cfg.scene_analysis
        cursor = 0
        with torch.no_grad():
            for start in tqdm(range(0, len(self.data), acfg.io_chunk_size), desc="Loading Gaussians", unit="chunk"):
                stop = min(len(self.data), start + acfg.io_chunk_size)
                block = self._matrix(slice(start, stop), self.all_required_names)
                mask = self._valid_mask(block)
                b = block[mask]
                m = len(b)
                if not m:
                    continue
                end = cursor + m
                xyz[cursor:end].copy_(torch.from_numpy(np.ascontiguousarray(b[:, 0:3])).to(device))
                rot = torch.from_numpy(np.ascontiguousarray(b[:, 3:7])).to(device)
                rot = torch.nn.functional.normalize(rot, dim=1)
                root_xyzw = Rotation.from_matrix(self.root_rotation).as_quat()
                root_q = torch.tensor(
                    [root_xyzw[3], root_xyzw[0], root_xyzw[1], root_xyzw[2]],
                    dtype=torch.float32,
                    device=device,
                )
                rw, rv = root_q[0], root_q[1:]
                qw, qv = rot[:, :1], rot[:, 1:]
                composed = torch.cat(
                    [
                        rw * qw - torch.sum(rv * qv, dim=1, keepdim=True),
                        rw * qv + qw * rv + torch.linalg.cross(rv.expand_as(qv), qv),
                    ],
                    dim=1,
                )
                rotations[cursor:end].copy_(
                    torch.nn.functional.normalize(composed, dim=1)
                )
                scales[cursor:end].copy_(
                    self.root_scale
                    * torch.exp(
                        torch.from_numpy(np.ascontiguousarray(b[:, 7:10])).to(device)
                    )
                )
                op = self._opacity(b[:, 10], self.analysis.opacity_storage)[:, None]
                opacity[cursor:end].copy_(torch.from_numpy(np.ascontiguousarray(op)).to(device))
                dc = b[:, 11:14, None]
                rest = b[:, 14:].reshape(m, 3, coeff - 1)
                features = np.concatenate([dc, rest], axis=2).transpose(0, 2, 1)
                shs[cursor:end].copy_(torch.from_numpy(np.ascontiguousarray(features)).to(device))
                cursor = end
                del block, b
        if cursor != n:
            raise RuntimeError(f"internal valid-count mismatch: expected {n}, loaded {cursor}")
        requested = self.available_sh_degree if self.cfg.render.sh_degree == "auto" else self.cfg.render.sh_degree
        if requested > self.available_sh_degree:
            raise ValueError(f"requested SH degree {requested}, PLY only has degree {self.available_sh_degree}")
        return GaussianTensors(xyz, scales, rotations, opacity, shs, int(requested))
