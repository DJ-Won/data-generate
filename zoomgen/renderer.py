from __future__ import annotations

import math

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .camera import CameraFrame, projection_matrix
from .config import GeneratorConfig
from .scene import GaussianTensors


class GaussianRenderer:
    def __init__(self, gaussians: GaussianTensors, cfg: GeneratorConfig):
        self.g = gaussians
        self.cfg = cfg
        self.device = gaussians.xyz.device
        self.means2d = torch.zeros_like(gaussians.xyz)
        self.alpha_colors = torch.ones((len(gaussians.xyz), 3), device=self.device)

    def _rasterizer(self, camera: CameraFrame, width: int, height: int, background: tuple[float, float, float]):
        # Preview cameras may carry full-resolution intrinsics; rescale here.
        sx, sy = width / self.cfg.video.width, height / self.cfg.video.height
        cam = CameraFrame(camera.position, camera.target, camera.c2w, camera.w2c,
                          camera.fx * sx, camera.fy * sy, camera.cx * sx, camera.cy * sy,
                          camera.fov_x, camera.fov_y, camera.near, camera.far,
                          camera.camera_center_offset)
        view = torch.tensor(cam.w2c, dtype=torch.float32, device=self.device).transpose(0, 1)
        proj = torch.tensor(projection_matrix(cam, width, height), dtype=torch.float32,
                            device=self.device).transpose(0, 1)
        settings = GaussianRasterizationSettings(
            image_height=height,
            image_width=width,
            tanfovx=width / (2.0 * cam.fx),
            tanfovy=height / (2.0 * cam.fy),
            bg=torch.tensor(background, dtype=torch.float32, device=self.device),
            scale_modifier=1.0,
            viewmatrix=view,
            projmatrix=view @ proj,
            sh_degree=self.g.sh_degree,
            campos=torch.tensor(cam.position, dtype=torch.float32, device=self.device),
            prefiltered=False,
            debug=False,
            antialiasing=self.cfg.render.antialiasing,
        )
        return GaussianRasterizer(settings)

    @torch.inference_mode()
    def render(
        self,
        camera: CameraFrame,
        width: int | None = None,
        height: int | None = None,
        alpha_only: bool = False,
        return_radii: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, torch.Tensor]:
        width = width or self.cfg.video.width
        height = height or self.cfg.video.height
        bg = (0.0, 0.0, 0.0) if alpha_only else self.cfg.render.background_rgb
        rasterizer = self._rasterizer(camera, width, height, bg)
        kwargs = dict(
            means3D=self.g.xyz,
            means2D=self.means2d,
            opacities=self.g.opacity,
            scales=self.g.scales,
            rotations=self.g.rotations,
        )
        if alpha_only:
            kwargs["colors_precomp"] = self.alpha_colors
        else:
            kwargs["shs"] = self.g.shs
        result = rasterizer(**kwargs)
        image = result[0] if isinstance(result, tuple) else result
        array = image.clamp_min(0).permute(1, 2, 0).contiguous().cpu().numpy()
        output = array[..., 0] if alpha_only else array
        if not return_radii:
            return output
        if not isinstance(result, tuple) or len(result) < 2:
            raise RuntimeError("Gaussian rasterizer did not return screen-space radii")
        return output, result[1].detach()
