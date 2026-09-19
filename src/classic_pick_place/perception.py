from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CupDetection:
    position_world: np.ndarray
    pixel_uv: tuple[float, float]
    confidence: float
    visible_pixels: int


class RgbdCupDetector:
    """Classical RGB-D detector for the blue cup used by the MuJoCo scene.

    Only rendered RGB and depth are used to estimate position. Simulator body
    coordinates are deliberately not consulted by the detector.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera_name: str = "perception",
        width: int = 640,
        height: int = 480,
        cup_half_height: float = 0.055,
    ):
        self.model = model
        self.width = width
        self.height = height
        self.cup_half_height = cup_half_height
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            raise ValueError(f"Camera {camera_name!r} was not found")
        self.camera_name = camera_name
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def close(self) -> None:
        self.renderer.close()

    def capture(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(data, camera=self.camera_name)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def _largest_blue_component(self, rgb: np.ndarray) -> np.ndarray:
        color = rgb.astype(np.float32)
        red, green, blue = color[..., 0], color[..., 1], color[..., 2]
        # Absolute channel gaps reject the blue-gray floor tiles while keeping
        # both the bright handle and shaded cylindrical cup surface.
        mask = (blue > 120.0) & ((blue - green) > 32.0) & ((blue - red) > 80.0)
        mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
        mask = ndimage.binary_closing(mask, structure=np.ones((5, 5)))
        labels, count = ndimage.label(mask)
        if count == 0:
            raise RuntimeError("No blue cup was detected in the RGB image")
        sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
        label = int(np.argmax(sizes)) + 1
        component = labels == label
        if int(component.sum()) < 80:
            raise RuntimeError("Blue detection is too small to be a reliable cup observation")
        return component

    def _deproject(self, data: mujoco.MjData, u: float, v: float, depth: float) -> np.ndarray:
        fovy = np.deg2rad(float(self.model.cam_fovy[self.camera_id]))
        focal = 0.5 * self.height / np.tan(0.5 * fovy)
        x = (u - 0.5 * (self.width - 1)) * depth / focal
        y = -(v - 0.5 * (self.height - 1)) * depth / focal
        point_camera = np.array([x, y, -depth])
        rotation = data.cam_xmat[self.camera_id].reshape(3, 3)
        return data.cam_xpos[self.camera_id] + rotation @ point_camera

    def detect(
        self,
        data: mujoco.MjData,
        *,
        output_dir: Path | None = None,
    ) -> CupDetection:
        rgb, depth = self.capture(data)
        mask = self._largest_blue_component(rgb)
        rows, cols = np.nonzero(mask)
        # The cup has a dark hollow center, so the median of visible blue
        # pixels is biased toward the illuminated rim. The bounding-box center
        # is a better estimate of the circular body center in the top view.
        u = 0.5 * float(cols.min() + cols.max())
        v = 0.5 * float(rows.min() + rows.max())
        z = float(np.median(depth[mask]))
        top_position = self._deproject(data, u, v, z)
        position = top_position.copy()
        position[2] -= self.cup_half_height

        pixels = int(mask.sum())
        detection = CupDetection(
            position_world=position,
            pixel_uv=(u, v),
            confidence=float(min(1.0, pixels / 900.0)),
            visible_pixels=pixels,
        )

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(output_dir / "camera_rgb.png", rgb)
            overlay = rgb.astype(np.float32)
            overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255.0, 230.0, 0.0])
            ui, vi = int(round(u)), int(round(v))
            overlay[max(0, vi - 8): vi + 9, max(0, ui - 1): ui + 2] = [255, 30, 30]
            overlay[max(0, vi - 1): vi + 2, max(0, ui - 8): ui + 9] = [255, 30, 30]
            imageio.imwrite(output_dir / "camera_detection.png", np.clip(overlay, 0, 255).astype(np.uint8))
        return detection

