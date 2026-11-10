"""
Frame input.

Real footage arrives either as a video file (OpenCV or imageio) or, for a
controlled experiment, as a directory of images.  This module hides that
difference behind a single iterator so that the rest of the pipeline never has
to care where the frames came from, and so the pipeline stays runnable on a
machine where OpenCV is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class FrameStream:
    """A source of frames with a known frame rate and origin."""

    source: str
    fps: float
    n_frames: int | None
    width: int | None = None
    height: int | None = None

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        raise NotImplementedError


class ImageSequence(FrameStream):
    """Frames read from a directory of image files, sorted by name."""

    def __init__(self, directory: str | Path, fps: float = 200.0) -> None:
        self.path = Path(directory)
        if not self.path.is_dir():
            raise FileNotFoundError(f"not a directory: {self.path}")
        files = sorted(p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise FileNotFoundError(f"no image files found in {self.path}")
        self.files = files
        first = _read_image(files[0])
        super().__init__(
            source=str(self.path),
            fps=float(fps),
            n_frames=len(files),
            width=first.shape[1],
            height=first.shape[0],
        )

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        for i, path in enumerate(self.files):
            yield i, _read_image(path)

    def timestamp(self, index: int) -> float:
        return index / self.fps


class VideoFile(FrameStream):
    """Frames read from a video file using OpenCV, falling back to imageio."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"no such video: {self.path}")
        self._backend = None
        self._reader = None

        try:  # prefer OpenCV: it reports the frame rate reliably
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(self.path))
            if cap.isOpened():
                self._backend = "opencv"
                self._reader = cap
                fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
                cap.release()
                super().__init__(source=str(self.path), fps=fps, n_frames=n, width=w, height=h)
                return
        except ImportError:
            pass

        try:  # fall back to imageio
            import imageio.v3 as iio  # type: ignore

            meta = iio.immeta(str(self.path), plugin="pyav")
            fps = float(meta.get("fps", 25.0))
            self._backend = "imageio"
            self._reader = iio
            super().__init__(source=str(self.path), fps=fps, n_frames=meta.get("nframes"))
            return
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "could not open the video: install opencv-python-headless or imageio-ffmpeg"
            ) from exc

    def frames(self) -> Iterator[tuple[int, np.ndarray]]:
        if self._backend == "opencv":
            import cv2  # type: ignore

            cap = cv2.VideoCapture(str(self.path))
            i = 0
            try:
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    yield i, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    i += 1
            finally:
                cap.release()
        else:
            iio = self._reader
            for i, frame in enumerate(iio.imiter(str(self.path), plugin="pyav")):
                yield i, np.asarray(frame)

    def timestamp(self, index: int) -> float:
        return index / self.fps


def open_source(source: str | Path, fps: float = 200.0) -> FrameStream:
    """Open either a directory of images or a video file."""
    path = Path(source)
    if path.is_dir():
        return ImageSequence(path, fps=fps)
    return VideoFile(path)


def _read_image(path: Path) -> np.ndarray:
    """Read an image as an RGB uint8 array, using Pillow or OpenCV."""
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as im:
            return np.asarray(im.convert("RGB"), dtype=np.uint8)
    except ImportError:  # pragma: no cover
        import cv2  # type: ignore

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"could not read {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
