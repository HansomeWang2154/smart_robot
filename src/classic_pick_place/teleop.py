from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TeleopCommand:
    twist: np.ndarray = field(default_factory=lambda: np.zeros(6))
    active: bool = False
    toggle_gripper: bool = False
    save_episode: bool = False
    discard_episode: bool = False
    reset: bool = False
    quit: bool = False


class PygameTeleopDevice:
    """Keyboard or SDL gamepad input with edge-triggered episode controls."""

    def __init__(self, device: str, *, headless: bool = False):
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Teleoperation needs pygame; install with `pip install -e '.[teleop]'`."
            ) from exc
        self.pg = pygame
        self.device = device
        self.headless = headless
        pygame.init()
        pygame.joystick.init()
        self.screen = None
        if not headless:
            self.screen = pygame.display.set_mode((960, 544))
            pygame.display.set_caption("Panda MuJoCo teleoperation")
        self.joystick = None
        if device == "gamepad":
            if pygame.joystick.get_count() == 0:
                raise RuntimeError("No SDL gamepad found. Use --device keyboard or connect a gamepad.")
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Using gamepad: {self.joystick.get_name()}")
        self.previous_buttons: dict[int, bool] = {}

    @staticmethod
    def _deadzone(value: float, threshold: float = 0.12) -> float:
        if abs(value) <= threshold:
            return 0.0
        return float(np.sign(value) * (abs(value) - threshold) / (1.0 - threshold))

    def _button_edge(self, index: int) -> bool:
        assert self.joystick is not None
        if index >= self.joystick.get_numbuttons():
            return False
        current = bool(self.joystick.get_button(index))
        edge = current and not self.previous_buttons.get(index, False)
        self.previous_buttons[index] = current
        return edge

    def poll(self) -> TeleopCommand:
        pg = self.pg
        quit_requested = False
        key_edges: set[int] = set()
        for event in pg.event.get():
            if event.type == pg.QUIT:
                quit_requested = True
            elif event.type == pg.KEYDOWN:
                key_edges.add(event.key)
        if self.device == "gamepad":
            return self._poll_gamepad(quit_requested)
        return self._poll_keyboard(quit_requested, key_edges)

    def _poll_gamepad(self, quit_requested: bool) -> TeleopCommand:
        assert self.joystick is not None
        joystick = self.joystick
        axis = lambda i: self._deadzone(joystick.get_axis(i)) if i < joystick.get_numaxes() else 0.0
        hat = joystick.get_hat(0) if joystick.get_numhats() else (0, 0)
        # SDL/Xbox layout: left stick translation, triggers vertical motion,
        # right stick roll/pitch, D-pad horizontal yaw. Hold LB as deadman.
        left_x, left_y = axis(0), axis(1)
        right_x = axis(3) if joystick.get_numaxes() >= 5 else axis(2)
        right_y = axis(4) if joystick.get_numaxes() >= 5 else axis(3)
        left_trigger = 0.5 * (axis(2) + 1.0) if joystick.get_numaxes() >= 6 else 0.0
        right_trigger = 0.5 * (axis(5) + 1.0) if joystick.get_numaxes() >= 6 else 0.0
        active = bool(joystick.get_button(4)) if joystick.get_numbuttons() > 4 else True
        twist = np.array(
            [-left_y, -left_x, right_trigger - left_trigger, -right_y, right_x, hat[0]],
            dtype=float,
        )
        return TeleopCommand(
            twist=twist,
            active=active,
            toggle_gripper=self._button_edge(0),       # A / Cross
            reset=self._button_edge(2),                # X / Square
            quit=quit_requested or self._button_edge(3),  # Y / Triangle
            discard_episode=self._button_edge(6),      # Back / Create
            save_episode=self._button_edge(7),         # Start / Options
        )

    def _poll_keyboard(
        self, quit_requested: bool, edges: set[int]
    ) -> TeleopCommand:
        pg = self.pg
        keys = pg.key.get_pressed()
        twist = np.array(
            [
                float(keys[pg.K_w]) - float(keys[pg.K_s]),
                float(keys[pg.K_a]) - float(keys[pg.K_d]),
                float(keys[pg.K_e]) - float(keys[pg.K_q]),
                float(keys[pg.K_u]) - float(keys[pg.K_o]),
                float(keys[pg.K_i]) - float(keys[pg.K_k]),
                float(keys[pg.K_j]) - float(keys[pg.K_l]),
            ]
        )
        return TeleopCommand(
            twist=twist,
            active=bool(keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]),
            toggle_gripper=pg.K_SPACE in edges,
            save_episode=pg.K_RETURN in edges,
            discard_episode=pg.K_BACKSPACE in edges,
            reset=pg.K_r in edges,
            quit=quit_requested or pg.K_ESCAPE in edges,
        )

    def show(self, rgb: np.ndarray, status: str) -> None:
        if self.screen is None:
            return
        pg = self.pg
        surface = pg.surfarray.make_surface(np.swapaxes(rgb, 0, 1))
        self.screen.blit(pg.transform.smoothscale(surface, self.screen.get_size()), (0, 0))
        font = pg.font.SysFont("Consolas", 20)
        background = pg.Surface((self.screen.get_width(), 34), pg.SRCALPHA)
        background.fill((0, 0, 0, 170))
        self.screen.blit(background, (0, 0))
        self.screen.blit(font.render(status, True, (245, 245, 245)), (10, 7))
        pg.display.flip()

    def close(self) -> None:
        self.pg.quit()

