from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .config import VideoConfig


class FFmpegWriter:
    def __init__(self, path: Path, cfg: VideoConfig):
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise RuntimeError("FFmpeg was not found; run inside the configured conda environment")
        self.cfg = cfg
        cmd = [
            exe, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{cfg.width}x{cfg.height}", "-framerate", str(cfg.fps),
            "-i", "-", "-an", "-c:v", cfg.codec, "-preset", cfg.preset,
            "-crf", str(cfg.crf), "-pix_fmt", cfg.pixel_format,
            "-movflags", "+faststart", str(path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.count = 0

    def append(self, rgb_u8: np.ndarray) -> None:
        if rgb_u8.shape != (self.cfg.height, self.cfg.width, 3) or rgb_u8.dtype != np.uint8:
            raise ValueError("FFmpegWriter expects HxWx3 uint8 RGB frames")
        assert self.proc.stdin is not None
        try:
            self.proc.stdin.write(rgb_u8.tobytes())
        except BrokenPipeError as exc:
            err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
            raise RuntimeError(f"FFmpeg stopped while encoding: {err}") from exc
        self.count += 1

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        err = self.proc.stderr.read().decode("utf-8", "replace") if self.proc.stderr else ""
        code = self.proc.wait()
        if code:
            raise RuntimeError(f"FFmpeg exited with status {code}: {err}")

    def abort(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


def probe_video(path: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot reopen encoded video: {path}")
    result = {
        "frame_count": int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),
    }
    cap.release()
    return result
