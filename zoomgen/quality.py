from __future__ import annotations

import math

import numpy as np
import torch

from .camera import CameraFrame
from .scene import GaussianTensors


PROJECTION_CANDIDATE_LIMIT = 4096
DOMINANCE_CANDIDATE_LIMIT = 64
SUPPORT_SIGMA = 3.0
MIN_CONTRIBUTION = 0.01
ANTIALIAS_VARIANCE_PX2 = 0.3


def _empty_metrics(visible_count: int, width: int, height: int) -> dict:
    return {
        "method": "rasterizer_radii_plus_screen_covariance_ownership",
        "is_approximate": True,
        "projection_candidate_selection": "largest_positive_native_rasterizer_radii",
        "dominance_candidate_selection": "largest_opacity_times_projected_area",
        "pixel_denominator": "all_image_pixels",
        "image_pixel_count": int(width * height),
        "support_sigma": SUPPORT_SIGMA,
        "minimum_contribution": MIN_CONTRIBUTION,
        "projection_candidate_limit": PROJECTION_CANDIDATE_LIMIT,
        "projection_is_candidate_limited": bool(
            visible_count > PROJECTION_CANDIDATE_LIMIT),
        "dominance_candidate_limit": DOMINANCE_CANDIDATE_LIMIT,
        "visible_gaussian_count": int(visible_count),
        "projection_candidate_count": 0,
        "dominance_is_candidate_limited": False,
        "dominance_candidate_count": 0,
        "max_rasterizer_radius_px": 0.0,
        "max_projected_area_px2": 0.0,
        "max_projected_area_ratio": 0.0,
        "max_projected_area_gaussian_index": None,
        "max_projected_major_axis_px": 0.0,
        "max_projected_major_axis_to_image_diagonal_ratio": 0.0,
        "max_projected_major_axis_gaussian_index": None,
        "largest_single_gaussian_dominated_pixel_ratio": 0.0,
        "top5_gaussians_dominated_pixel_ratio": 0.0,
        "candidate_coverage_pixel_ratio": 0.0,
        "multi_layer_overlap_pixel_ratio": 0.0,
        "max_overlap_layer_count": 0,
        "top_dominant_gaussians": [],
    }


def _rotation_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    q = quaternions / np.maximum(norms, 1e-12)
    w, x, y, z = q.T
    matrices = np.empty((len(q), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[:, 0, 1] = 2.0 * (x * y - w * z)
    matrices[:, 0, 2] = 2.0 * (x * z + w * y)
    matrices[:, 1, 0] = 2.0 * (x * y + w * z)
    matrices[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[:, 1, 2] = 2.0 * (y * z - w * x)
    matrices[:, 2, 0] = 2.0 * (x * z - w * y)
    matrices[:, 2, 1] = 2.0 * (y * z + w * x)
    matrices[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


def _project_candidates(
    gaussians: GaussianTensors,
    indices: torch.Tensor,
    camera: CameraFrame,
) -> dict[str, np.ndarray]:
    indices = indices.to(device=gaussians.xyz.device, dtype=torch.long)
    xyz = gaussians.xyz.index_select(0, indices).detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    scales = gaussians.scales.index_select(0, indices).detach().cpu().numpy().astype(
        np.float64, copy=False
    )
    rotations = (
        gaussians.rotations.index_select(0, indices)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float64, copy=False)
    )
    opacity = (
        gaussians.opacity.index_select(0, indices)
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
        .astype(np.float64, copy=False)
    )
    gaussian_indices = indices.detach().cpu().numpy().astype(np.int64, copy=False)

    w2c = np.asarray(camera.w2c, dtype=np.float64)
    rotation_world_to_camera = w2c[:3, :3]
    camera_xyz = xyz @ rotation_world_to_camera.T + w2c[:3, 3]
    z = camera_xyz[:, 2]

    rotation_gaussian = _rotation_matrices_wxyz(rotations)
    scaled_rotation = rotation_gaussian * scales[:, None, :]
    camera_axes = np.einsum(
        "ij,njk->nik", rotation_world_to_camera, scaled_rotation
    )

    safe_z = np.maximum(z, 1e-12)
    x_over_z = camera_xyz[:, 0] / safe_z
    y_over_z = camera_xyz[:, 1] / safe_z
    row_u = (camera.fx / safe_z)[:, None] * (
        camera_axes[:, 0, :] - x_over_z[:, None] * camera_axes[:, 2, :]
    )
    row_v = (camera.fy / safe_z)[:, None] * (
        camera_axes[:, 1, :] - y_over_z[:, None] * camera_axes[:, 2, :]
    )
    covariance_00 = np.sum(row_u * row_u, axis=1) + ANTIALIAS_VARIANCE_PX2
    covariance_01 = np.sum(row_u * row_v, axis=1)
    covariance_11 = np.sum(row_v * row_v, axis=1) + ANTIALIAS_VARIANCE_PX2
    determinant = covariance_00 * covariance_11 - covariance_01 * covariance_01
    trace = covariance_00 + covariance_11
    discriminant = np.sqrt(
        np.maximum(0.0, trace * trace - 4.0 * determinant)
    )
    eigenvalue_major = np.maximum(0.0, 0.5 * (trace + discriminant))
    eigenvalue_minor = np.maximum(0.0, 0.5 * (trace - discriminant))
    semi_major = SUPPORT_SIGMA * np.sqrt(eigenvalue_major)
    semi_minor = SUPPORT_SIGMA * np.sqrt(eigenvalue_minor)
    area = math.pi * semi_major * semi_minor
    center_x = camera.fx * x_over_z + camera.cx
    center_y = camera.fy * y_over_z + camera.cy

    valid = (
        np.isfinite(center_x)
        & np.isfinite(center_y)
        & np.isfinite(area)
        & np.isfinite(determinant)
        & (z > camera.near)
        & (determinant > 1e-12)
    )
    return {
        "gaussian_index": gaussian_indices[valid],
        "opacity": np.clip(opacity[valid], 0.0, 1.0),
        "center_x": center_x[valid],
        "center_y": center_y[valid],
        "covariance_00": covariance_00[valid],
        "covariance_01": covariance_01[valid],
        "covariance_11": covariance_11[valid],
        "determinant": determinant[valid],
        "semi_major": semi_major[valid],
        "semi_minor": semi_minor[valid],
        "area": area[valid],
    }


def _ownership_metrics(
    projected: dict[str, np.ndarray],
    width: int,
    height: int,
) -> dict:
    count = len(projected["gaussian_index"])
    if count == 0:
        return {
            "dominance_candidate_count": 0,
            "largest_single_gaussian_dominated_pixel_ratio": 0.0,
            "top5_gaussians_dominated_pixel_ratio": 0.0,
            "candidate_coverage_pixel_ratio": 0.0,
            "multi_layer_overlap_pixel_ratio": 0.0,
            "max_overlap_layer_count": 0,
            "top_dominant_gaussians": [],
        }

    integrated_contribution = projected["opacity"] * projected["area"]
    candidate_count = min(DOMINANCE_CANDIDATE_LIMIT, count)
    order = np.argsort(integrated_contribution)[-candidate_count:][::-1]
    best_weight = np.zeros((height, width), dtype=np.float32)
    best_owner = np.full((height, width), -1, dtype=np.int32)
    overlap_count = np.zeros((height, width), dtype=np.uint16)

    for owner, projected_index in enumerate(order):
        center_x = float(projected["center_x"][projected_index])
        center_y = float(projected["center_y"][projected_index])
        covariance_00 = float(projected["covariance_00"][projected_index])
        covariance_01 = float(projected["covariance_01"][projected_index])
        covariance_11 = float(projected["covariance_11"][projected_index])
        determinant = float(projected["determinant"][projected_index])
        opacity = float(projected["opacity"][projected_index])

        extent_x = SUPPORT_SIGMA * math.sqrt(max(covariance_00, 0.0))
        extent_y = SUPPORT_SIGMA * math.sqrt(max(covariance_11, 0.0))
        x0 = max(0, int(math.floor(center_x - extent_x)))
        x1 = min(width, int(math.ceil(center_x + extent_x)) + 1)
        y0 = max(0, int(math.floor(center_y - extent_y)))
        y1 = min(height, int(math.ceil(center_y + extent_y)) + 1)
        if x0 >= x1 or y0 >= y1 or opacity < MIN_CONTRIBUTION:
            continue

        xs = np.arange(x0, x1, dtype=np.float64) + 0.5 - center_x
        ys = np.arange(y0, y1, dtype=np.float64) + 0.5 - center_y
        dx, dy = np.meshgrid(xs, ys)
        mahalanobis = (
            covariance_11 * dx * dx
            - 2.0 * covariance_01 * dx * dy
            + covariance_00 * dy * dy
        ) / determinant
        weight = opacity * np.exp(-0.5 * mahalanobis)
        supported = (
            (mahalanobis <= SUPPORT_SIGMA * SUPPORT_SIGMA)
            & (weight >= MIN_CONTRIBUTION)
        )
        overlap_view = overlap_count[y0:y1, x0:x1]
        overlap_view[supported] += 1

        best_view = best_weight[y0:y1, x0:x1]
        owner_view = best_owner[y0:y1, x0:x1]
        update = supported & (weight > best_view)
        best_view[update] = weight[update].astype(np.float32, copy=False)
        owner_view[update] = owner

    pixel_count = width * height
    owned = best_owner >= 0
    counts = np.bincount(
        best_owner[owned], minlength=candidate_count
    ) if np.any(owned) else np.zeros(candidate_count, dtype=np.int64)
    dominant_order = np.argsort(counts)[::-1]
    nonzero_order = [index for index in dominant_order if counts[index] > 0]
    top_five = nonzero_order[:5]
    top_entries = []
    for owner in top_five:
        projected_index = int(order[owner])
        dominated_count = int(counts[owner])
        top_entries.append(
            {
                "gaussian_index": int(projected["gaussian_index"][projected_index]),
                "dominated_pixel_count": dominated_count,
                "dominated_pixel_ratio": float(dominated_count / pixel_count),
                "projected_area_px2": float(projected["area"][projected_index]),
                "projected_major_axis_px": float(
                    2.0 * projected["semi_major"][projected_index]
                ),
                "opacity": float(projected["opacity"][projected_index]),
            }
        )

    largest_count = int(counts[dominant_order[0]]) if len(dominant_order) else 0
    return {
        "dominance_candidate_count": int(candidate_count),
        "dominance_is_candidate_limited": bool(count > candidate_count),
        "largest_single_gaussian_dominated_pixel_ratio": float(
            largest_count / pixel_count
        ),
        "top5_gaussians_dominated_pixel_ratio": float(
            sum(int(counts[index]) for index in top_five) / pixel_count
        ),
        "candidate_coverage_pixel_ratio": float(np.mean(owned)),
        "multi_layer_overlap_pixel_ratio": float(np.mean(overlap_count >= 2)),
        "max_overlap_layer_count": int(overlap_count.max(initial=0)),
        "top_dominant_gaussians": top_entries,
    }


def screen_space_quality_metrics(
    gaussians: GaussianTensors,
    camera: CameraFrame,
    radii: torch.Tensor,
    width: int,
    height: int,
) -> dict:
    radii = radii.detach().reshape(-1)
    visible_count = int(torch.count_nonzero(radii > 0).item())
    metrics = _empty_metrics(visible_count, width, height)
    if visible_count == 0:
        return metrics

    candidate_count = min(PROJECTION_CANDIDATE_LIMIT, visible_count)
    candidate_radii, candidate_indices = torch.topk(
        radii, k=candidate_count, largest=True, sorted=True
    )
    projected = _project_candidates(gaussians, candidate_indices, camera)
    projected_count = len(projected["gaussian_index"])
    if projected_count == 0:
        return metrics

    max_area_local = int(np.argmax(projected["area"]))
    max_major_local = int(np.argmax(projected["semi_major"]))
    image_pixel_count = width * height
    image_diagonal = math.hypot(width, height)
    metrics.update(
        {
            "projection_candidate_count": int(projected_count),
            "max_rasterizer_radius_px": float(candidate_radii[0].item()),
            "max_projected_area_px2": float(projected["area"][max_area_local]),
            "max_projected_area_ratio": float(
                projected["area"][max_area_local] / image_pixel_count
            ),
            "max_projected_area_gaussian_index": int(
                projected["gaussian_index"][max_area_local]
            ),
            "max_projected_major_axis_px": float(
                2.0 * projected["semi_major"][max_major_local]
            ),
            "max_projected_major_axis_to_image_diagonal_ratio": float(
                2.0 * projected["semi_major"][max_major_local] / image_diagonal
            ),
            "max_projected_major_axis_gaussian_index": int(
                projected["gaussian_index"][max_major_local]
            ),
        }
    )
    metrics.update(_ownership_metrics(projected, width, height))
    return metrics
