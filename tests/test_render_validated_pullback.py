from types import SimpleNamespace

import numpy as np

from render_validated_zoom_dataset import _adapt_traversal_camera
from zoomgen.camera import CameraFrame


def _frame(position=(0.0, 0.0, -1.0), forward=(0.0, 0.0, 1.0)):
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = np.eye(3, dtype=np.float64)
    c2w[:3, 2] = np.asarray(forward, dtype=np.float64)
    c2w[:3, 3] = np.asarray(position, dtype=np.float64)
    return CameraFrame(
        c2w[:3, 3].copy(),
        c2w[:3, 3] + c2w[:3, 2],
        c2w,
        np.linalg.inv(c2w),
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.01,
        100.0,
        np.zeros(3),
    )


def _scene(
    center=(0.0, 0.0, 3.0),
    lower=(-10.0, -10.0, -10.0),
    upper=(10.0, 10.0, 10.0),
):
    return SimpleNamespace(
        analysis=SimpleNamespace(
            center=np.array(center),
            radius=2.0,
            aabb_min=np.array(lower),
            aabb_max=np.array(upper),
        )
    )


def _pullback(**updates):
    values = {
        "enabled": True,
        "ratio": 0.5,
        "max_distance_ratio": 0.75,
        "boundary_margin_ratio": 0.02,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_pullback_moves_away_and_preserves_camera_axes():
    frame = _frame()
    adjusted, details = _adapt_traversal_camera(
        frame, _scene(), "object", _pullback()
    )

    assert adjusted.position[2] < frame.position[2]
    assert np.linalg.norm(adjusted.position - np.array([0.0, 0.0, 3.0])) > 4.0
    np.testing.assert_allclose(adjusted.c2w[:3, :3], frame.c2w[:3, :3])
    np.testing.assert_allclose(adjusted.w2c @ adjusted.c2w, np.eye(4), atol=1e-12)
    assert details["applied_distance"] > 0.0


def test_interior_pullback_is_limited_before_aabb_boundary():
    frame = _frame()
    adjusted, details = _adapt_traversal_camera(
        frame, _scene(upper=(10.0, 10.0, 10.0), lower=(-10.0, -10.0, -2.0)),
        "interior", _pullback()
    )

    np.testing.assert_allclose(adjusted.position[2], -1.98)
    assert details["limited_by_boundary"] is True


def test_pullback_skips_when_view_reversal_points_toward_scene_center():
    frame = _frame(position=(0.0, 0.0, 1.0))
    adjusted, details = _adapt_traversal_camera(
        frame, _scene(center=(0.0, 0.0, 0.0)), "object", _pullback()
    )

    np.testing.assert_allclose(adjusted.position, frame.position)
    assert details["skipped_toward_scene_center"] is True
