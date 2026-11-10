"""Camera model, triangulation, wicket geometry and the decision zones."""

from __future__ import annotations

import numpy as np
import pytest

from drs import physics
from drs.config import BALL_RADIUS, STUMP_HEIGHT, WICKET_HALF_WIDTH
from drs.geometry import (
    Camera,
    Wicket,
    broadcast_single_camera,
    default_stereo_rig,
    intersect_rays,
    triangulate,
    triangulate_sequence,
)


# --------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------


def test_projection_and_back_projection_are_consistent():
    cam = broadcast_single_camera()
    X = np.array([[-6.0, 0.12, 0.52], [-18.0, -0.1, 2.0], [0.0, 0.0, 0.7]])
    uv, depth = cam.project(X)
    assert np.all(depth > 0)

    origins, directions = cam.ray(uv)
    for i in range(X.shape[0]):
        s = float(np.dot(X[i] - origins[i], directions[i]))
        closest = origins[i] + s * directions[i]
        assert np.linalg.norm(closest - X[i]) < 1e-9


def test_camera_centre_maps_to_its_own_origin():
    cam = default_stereo_rig()[0]
    assert np.allclose(cam.position, -cam.R.T @ cam.t)


def test_a_pixel_defines_a_ray_not_a_point():
    """The depth ambiguity at the heart of the project."""
    cam = broadcast_single_camera()
    X = np.array([[-6.0, 0.0, 0.5]])
    uv, _ = cam.project(X)
    origins, directions = cam.ray(uv)
    # Two very different depths give exactly the same pixel.
    near = origins[0] + 5.0 * directions[0]
    far = origins[0] + 40.0 * directions[0]
    uv_near, _ = cam.project(near.reshape(1, 3))
    uv_far, _ = cam.project(far.reshape(1, 3))
    assert np.allclose(uv_near, uv_far, atol=1e-6)


def test_parallel_view_direction_and_up_is_rejected():
    with pytest.raises(ValueError):
        Camera.look_at((0, 0, 0), (0, 0, 5), up=(0, 0, 1))


def test_in_frame_reports_off_screen_points():
    cam = broadcast_single_camera()
    uv, _ = cam.project(np.array([[0.0, 0.0, 0.7], [0.0, 100.0, 0.7]]))
    flags = cam.in_frame(uv)
    assert flags[0] and not flags[1]


# --------------------------------------------------------------------------
# Triangulation
# --------------------------------------------------------------------------


def test_stereo_triangulation_recovers_points_exactly_when_noiseless():
    cams = default_stereo_rig()
    traj = physics.simulate_delivery(physics.preset("good_length_middle"))
    t = np.linspace(0.0, traj.duration, 15)
    X = traj.position(t)
    pixels = np.stack([c.project(X)[0] for c in cams], axis=0)
    recovered = triangulate_sequence(cams, pixels)
    assert np.abs(recovered - X).max() < 1e-9


def test_triangulation_needs_one_pixel_per_camera():
    cams = default_stereo_rig()
    with pytest.raises(ValueError):
        triangulate(cams, np.zeros((1, 2)))


def test_depth_is_the_worst_conditioned_coordinate():
    """A small pixel error must hurt depth far more than the lateral axes."""
    rng = np.random.default_rng(0)
    cams = default_stereo_rig()
    traj = physics.simulate_delivery(physics.preset("good_length_middle"))
    t = np.linspace(0.0, traj.duration, 20)
    X = traj.position(t)
    pixels = np.stack([c.project(X)[0] for c in cams], axis=0)

    errors = []
    for _ in range(20):
        noisy = pixels + rng.normal(0.0, 0.5, pixels.shape)
        errors.append(np.sqrt(np.mean((triangulate_sequence(cams, noisy) - X) ** 2, axis=0)))
    mean_err = np.mean(errors, axis=0)
    assert mean_err[0] > 5 * mean_err[1]
    assert mean_err[0] > 5 * mean_err[2]


def test_intersect_rays_finds_the_common_point_of_intersecting_rays():
    origin = np.array([0.0, 0.0, 0.0])
    target = np.array([1.0, 2.0, -0.5])
    origins = np.array([origin, origin + np.array([0.0, 1.0, 0.0])])
    directions = np.array([target - origins[0], target - origins[1]])
    X, residual = intersect_rays(origins, directions)
    assert np.allclose(X, target, atol=1e-9)
    assert residual < 1e-9


# --------------------------------------------------------------------------
# Wicket geometry
# --------------------------------------------------------------------------


def test_distance_to_the_wicket_is_zero_inside_a_stump():
    w = Wicket()
    assert w.distance(np.array([[0.0, 0.0, 0.35]]))[0] == pytest.approx(0.0)


def test_distance_above_the_stumps_is_the_vertical_gap():
    w = Wicket()
    d = w.distance(np.array([[0.0, 0.0, STUMP_HEIGHT + 0.10]]))[0]
    # Just above the stump top but still inside a bail, so measure the bail.
    assert d <= 0.10


def test_ball_hits_only_within_a_ball_radius():
    w = Wicket()
    on_line = np.array([[0.0, 0.0, 0.35]])
    just_clear = np.array([[0.0, WICKET_HALF_WIDTH + 2.5 * BALL_RADIUS, 0.35]])
    assert w.hits(on_line)[0]
    assert not w.hits(just_clear)[0]
    assert w.clearance(just_clear)[0] > 0


def test_a_ball_grazing_the_outer_edge_is_the_umpires_call_boundary():
    w = Wicket()
    # The outer leg stump's surface sits at y = -half_width, so the ball's
    # centre touches it exactly one ball-radius further out.
    y_touch = -(WICKET_HALF_WIDTH + BALL_RADIUS)
    grazing = np.array([[0.0, y_touch, 0.35]])
    assert abs(float(w.clearance(grazing)[0])) < 1e-9


def test_lateral_zone_classification():
    w = Wicket()
    assert w.in_line(0.0)[0]
    assert w.outside_off(0.3)[0]
    assert w.outside_leg(-0.3)[0]
    assert not w.outside_off(-0.3)[0]
