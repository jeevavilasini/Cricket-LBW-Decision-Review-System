"""Image processing, tracking and the synthetic renderer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from drs import physics
from drs.detection import (
    Blob,
    ball_mask,
    detect_ball,
    detect_blobs,
    detect_sequence,
    redness,
    refine_centroid,
    rgb_to_hsv,
)
from drs.geometry import Camera, default_stereo_rig
from drs.simulate import build_ground_truth
from drs.tracking import (
    PixelTrack,
    pixel_rmse,
    reconstruct_on_plane,
    reject_outliers,
    track_pixels,
    triangulate_tracks,
)
from drs.video import ImageSequence, open_source

BALL_RGB = (200, 40, 35)


# --------------------------------------------------------------------------
# Colour conversion
# --------------------------------------------------------------------------


def test_pure_red_maps_to_hue_zero_in_opencv_units():
    hsv = rgb_to_hsv(np.array([[[255, 0, 0]]], dtype=float))
    assert hsv[0, 0, 0] == pytest.approx(0.0, abs=1e-9)
    assert hsv[0, 0, 1] == pytest.approx(255.0, abs=1e-9)
    assert hsv[0, 0, 2] == pytest.approx(255.0, abs=1e-9)


def test_greyscale_has_zero_saturation():
    grey = np.full((1, 1, 3), 128.0)
    hsv = rgb_to_hsv(grey)
    assert hsv[0, 0, 1] == pytest.approx(0.0)


def test_a_red_ball_survives_a_brightness_change():
    """The point of working in HSV: hue is stable under exposure changes."""
    dark = np.array([[[140, 30, 25]]], dtype=float)
    bright = np.array([[[255, 90, 80]]], dtype=float)
    h_dark = rgb_to_hsv(dark)[0, 0, 0]
    h_bright = rgb_to_hsv(bright)[0, 0, 0]
    assert abs(h_dark - h_bright) < 2.0
    assert ball_mask(dark)[0, 0] and ball_mask(bright)[0, 0]


def test_grass_is_not_masked_as_ball():
    grass = np.array([[[96, 118, 62], [138, 150, 96]]], dtype=float)
    assert not ball_mask(grass).any()
    assert np.all(redness(grass) < 20)


# --------------------------------------------------------------------------
# Blob detection
# --------------------------------------------------------------------------


def _scene(radius: float = 4.0, centre=(100.0, 60.0), size=(120, 200)):
    """A red disc on a green field: the simplest unambiguous test image."""
    img = np.zeros((size[0], size[1], 3), dtype=float)
    img[:, :] = (96, 118, 62)
    yy, xx = np.mgrid[0 : size[0], 0 : size[1]]
    d = np.hypot(xx - centre[0], yy - centre[1])
    m = d <= radius
    img[m] = BALL_RGB
    return img


def test_a_clean_disc_is_found_at_the_right_place():
    img = _scene(radius=5.0, centre=(100.0, 60.0))
    blobs = detect_blobs(img, min_area=4)
    assert len(blobs) == 1
    assert np.allclose(blobs[0].centroid, [100.0, 60.0], atol=1.0)
    assert blobs[0].area == pytest.approx(np.pi * 25, rel=0.2)


def test_an_empty_scene_yields_no_detection():
    img = _scene()
    img[:, :] = (96, 118, 62)
    assert detect_blobs(img, min_area=4) == []
    assert detect_ball(img) is None


def test_a_disc_split_in_two_by_grass_still_detects():
    img = _scene(radius=5.0)
    img[59:61, 70:130] = (96, 118, 62)  # a thin mower stripe across the ball
    blobs = detect_blobs(img, min_area=4)
    assert blobs, "the two remaining caps should still be detected"
    # Their union still straddles the true centre.
    ys = np.concatenate([np.full(b.area, b.centroid[1]) for b in blobs])
    assert abs(ys.mean() - 60.0) < 3.0


def test_subpixel_centroid_refinement_improves_accuracy():
    img = _scene(radius=6.0, centre=(101.3, 59.7))
    blob = detect_blobs(img, min_area=4)[0]
    coarse = blob.centroid.copy()
    fine = refine_centroid(img, blob, weight="redness")
    true = np.array([101.3, 59.7])
    assert np.linalg.norm(fine - true) <= np.linalg.norm(coarse - true) + 0.3


def test_the_gate_rejects_a_distant_blob():
    img = _scene(centre=(100.0, 60.0))
    expected = np.array([10.0, 10.0])
    assert detect_ball(img, expected=expected, gate_px=5.0) is None
    assert detect_ball(img, expected=expected, gate_px=500.0) is not None


def test_min_area_filters_specks():
    img = _scene(radius=1.0)
    assert detect_blobs(img, min_area=8) == []
    assert detect_blobs(img, min_area=1)


def test_detect_sequence_tracks_a_moving_disc():
    frames = [_scene(centre=(20.0 + 15 * i, 60.0)) for i in range(6)]
    blobs = detect_sequence(frames, min_area=4)
    assert len(blobs) == 6
    assert all(b is not None for b in blobs)
    xs = [b.centroid[0] for b in blobs]
    assert np.allclose(np.diff(xs), 15.0, atol=1.0)


def test_circularity_distinguishes_a_disc_from_a_streak():
    disc = Blob(centroid=np.array([0.0, 0.0]), area=50, bbox=(0, 0, 8, 8))
    streak = Blob(centroid=np.array([0.0, 0.0]), area=50, bbox=(0, 0, 60, 4))
    assert disc.circularity() > streak.circularity()


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


def test_a_single_outlier_is_gated_out():
    t = np.arange(6) / 50.0
    uv = np.column_stack([np.arange(6) * 4.0, np.full(6, 20.0)]).astype(float)
    uv[3] = [500.0, 500.0]
    track = PixelTrack(t, uv, np.ones(6, dtype=bool))
    gated = reject_outliers(track, max_speed_px_s=2000.0)
    assert gated.n_visible == 5
    assert not gated.visible[3]


def test_a_clean_track_is_untouched():
    t = np.arange(10) / 50.0
    uv = np.column_stack([np.arange(10) * 3.0, np.full(10, 20.0)]).astype(float)
    track = PixelTrack(t, uv, np.ones(10, dtype=bool))
    gated = reject_outliers(track, max_speed_px_s=2000.0)
    assert gated.n_visible == 10


def test_pixel_rmse_is_zero_for_a_perfect_track():
    t = np.arange(5) / 50.0
    uv = np.zeros((5, 2))
    track = PixelTrack(t, uv, np.ones(5, dtype=bool))
    assert pixel_rmse(track, uv) == pytest.approx(0.0)


def test_triangulating_two_tracks_recovers_the_trajectory():
    cams = default_stereo_rig()
    truth = build_ground_truth(physics.preset("good_length_middle"))
    t = np.linspace(0.0, truth.trajectory.duration, 25)
    X = truth.trajectory.position(t)

    tracks = []
    for cam in cams:
        uv, _ = cam.project(X)
        tracks.append(PixelTrack(t, uv, np.ones(t.size, dtype=bool)))

    obs = triangulate_tracks(tracks, cams)
    assert obs.n_visible == t.size
    assert np.abs(obs.points - X).max() < 1e-6
    assert obs.sigma > 0


def test_tracks_must_share_a_time_base():
    cams = default_stereo_rig()
    a = PixelTrack(np.arange(5) / 50.0, np.zeros((5, 2)), np.ones(5, dtype=bool))
    b = PixelTrack(np.arange(6) / 50.0, np.zeros((6, 2)), np.ones(6, dtype=bool))
    with pytest.raises(ValueError):
        triangulate_tracks([a, b], cams)


def test_triangulation_needs_two_cameras():
    track = PixelTrack(np.arange(5) / 50.0, np.zeros((5, 2)), np.ones(5, dtype=bool))
    with pytest.raises(ValueError):
        triangulate_tracks([track], [default_stereo_rig()[0]])


def test_single_camera_plane_reconstruction_lands_on_the_plane():
    cam = default_stereo_rig()[0]
    truth = build_ground_truth(physics.preset("good_length_middle"))
    t = np.linspace(0.0, truth.trajectory.duration, 15)
    X = truth.trajectory.position(t)
    uv, _ = cam.project(X)

    track = PixelTrack(t, uv, np.ones(t.size, dtype=bool))
    obs = reconstruct_on_plane(track, cam, plane_point=(0.0, 0.0, 0.0), plane_normal=(0.0, 0.0, 1.0))
    # Every reconstructed point must lie on the assumed plane, by construction.
    assert np.abs(obs.points[:, 2]).max() < 1e-9
    # And it is a poor model for a ball in flight.
    assert np.linalg.norm(obs.points - X, axis=1).mean() > 0.05


# --------------------------------------------------------------------------
# Rendering and reading back
# --------------------------------------------------------------------------


def test_rendered_frames_are_readable_and_contain_the_ball(tmp_path: Path):
    from drs.synthvideo import render_delivery

    cam = Camera.look_at((-25.0, -2.0, 3.2), (0.0, 0.0, 0.7), fov_deg=32.0, width=640, height=360, name="t")
    truth = build_ground_truth(physics.preset("good_length_middle"))
    seq = render_delivery(
        truth.trajectory, cam, tmp_path / "frames",
        fps=200.0, blur_subframes=3, noise_sigma=0.0, seed=0,
    )
    assert seq.n_frames >= 2
    assert (tmp_path / "frames" / "frame_0000.png").exists()

    stream = open_source(seq.directory, fps=seq.fps)
    frames = dict(stream.frames())
    assert len(frames) == seq.n_frames
    assert frames[0].shape == (360, 640, 3)

    # Check a mid-flight frame: the ball is well inside the frame and closer to
    # the camera, so it is a fair test of the detector rather than of the FOV.
    mid = seq.n_frames // 2
    assert seq.in_frame[mid]

    # The centroid of a blurred streak sits at the middle of the exposure, so the
    # achievable accuracy is set by how far the ball moves in one frame -- which
    # at 140 km/h dwarfs the ball's own size.
    blob = detect_ball(frames[mid], min_area=2)
    assert blob is not None
    displacement = float(np.linalg.norm(seq.true_pixels[mid] - seq.true_pixels[mid - 1]))
    assert np.linalg.norm(blob.centroid - seq.true_pixels[mid]) < max(5.0, 1.2 * displacement)


def test_rendering_purges_stale_frames(tmp_path: Path):
    from drs.synthvideo import render_delivery

    cam = Camera.look_at((-25.0, -2.0, 3.2), (0.0, 0.0, 0.7), fov_deg=32.0, width=200, height=120, name="t")
    truth = build_ground_truth(physics.preset("good_length_middle"))
    out = tmp_path / "frames"

    first = render_delivery(truth.trajectory, cam, out, fps=200.0, blur_subframes=1, noise_sigma=0.0)
    second = render_delivery(truth.trajectory, cam, out, fps=50.0, blur_subframes=1, noise_sigma=0.0)
    assert second.n_frames < first.n_frames
    assert len(list(out.glob("frame_*.png"))) == second.n_frames


def test_tracking_a_rendered_clip_finds_the_ball_in_most_frames(tmp_path: Path):
    from drs.synthvideo import render_delivery

    cams = default_stereo_rig()
    truth = build_ground_truth(physics.preset("good_length_middle"))
    seq = render_delivery(
        truth.trajectory, cams[0], tmp_path / "frames",
        fps=150.0, blur_subframes=2, noise_sigma=0.0,
    )
    stream = ImageSequence(seq.directory, fps=seq.fps)
    track, _ = track_pixels(stream.frames(), fps=stream.fps, gate_px=600.0, min_area=3)
    assert track.n_visible >= 0.8 * len(track)
    assert pixel_rmse(track, seq.true_pixels) < 3.0
