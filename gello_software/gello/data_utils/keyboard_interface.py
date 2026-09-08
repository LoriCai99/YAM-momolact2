import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import pygame

NORMAL = (128, 128, 128)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

KEY_START = pygame.K_s
# KEY_CONTINUE = pygame.K_c
KEY_QUIT_RECORDING = pygame.K_q
KEY_SAVE = pygame.K_a
KEY_DISCARD = pygame.K_b


class KBReset:
    def __init__(self):
        pygame.init()
        self._screen_width = 1500
        self._screen_height = 900
        self._screen = pygame.display.set_mode((self._screen_width, self._screen_height))
        pygame.display.set_caption("YAM Keyboard Interface")
        self._font = pygame.font.SysFont("monospace", 20)
        self._small_font = pygame.font.SysFont("monospace", 16)
        self._popup_text = ""
        self._popup_until = 0.0
        # The dashboard is a preview, not the control loop. Rendering three
        # 640x360 tiles via surfarray + smoothscale cost ~55 ms per call and
        # dragged the 30 Hz collection loop down to ~14 Hz (jerky arms) while
        # starving the arms' 250 Hz threads of the GIL. Render at most this
        # often; key events are still polled on every update() call.
        self.render_hz = 10.0
        self._last_render = 0.0
        self._set_color(NORMAL)
        self._saved = False

    # def update(self) -> str:
    #     pressed_last = self._get_pressed()
    #     if KEY_QUIT_RECORDING in pressed_last:
    #         self._set_color(RED)
    #         self._saved = False
    #         return "normal"

    #     if self._saved:
    #         return "save"

    #     if KEY_START in pressed_last:
    #         self._set_color(GREEN)
    #         self._saved = True
    #         return "start"

    #     self._set_color(NORMAL)
    #     return "normal"
    def update(self, dashboard_data: Optional[Dict[str, Any]] = None) -> str:
        pressed_last = self._get_pressed()
        if KEY_START in pressed_last:
            self._show_popup("Operation: start")
            if dashboard_data is not None:
                self._render_dashboard(dashboard_data)
            return "start"
        if KEY_SAVE in pressed_last:
            self._show_popup("Operation: save")
            if dashboard_data is not None:
                self._render_dashboard(dashboard_data)
            return "save"
        if KEY_DISCARD in pressed_last:
            self._show_popup("Operation: delete")
            if dashboard_data is not None:
                self._render_dashboard(dashboard_data)
            return "discard"
        if dashboard_data is not None:
            now = time.time()
            if now - self._last_render >= 1.0 / self.render_hz:
                self._render_dashboard(dashboard_data)
                self._last_render = now
        return "normal"

    def _get_pressed(self):
        pressed = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return [KEY_DISCARD]
            if event.type == pygame.KEYDOWN:
                pressed.append(event.key)
        return pressed

    def _set_color(self, color):
        self._screen.fill(color)
        pygame.display.flip()

    def _render_dashboard(self, dashboard_data: Dict[str, Any]) -> None:
        self._screen.fill((25, 25, 25))
        margin = 12
        header_h = 120
        camera_area_w = int(self._screen_width * 0.7)
        camera_area_h = self._screen_height - header_h - 2 * margin
        side_x = margin + camera_area_w + margin
        side_w = self._screen_width - side_x - margin

        self._draw_header(dashboard_data, margin, margin, self._screen_width - 2 * margin, header_h)

        cameras = dashboard_data.get("cameras", {})
        self._draw_camera_grid(
            cameras=cameras,
            x=margin,
            y=margin + header_h + margin,
            w=camera_area_w,
            h=camera_area_h,
        )
        self._draw_joint_panel(
            joint_positions=dashboard_data.get("joint_positions"),
            joint_velocities=dashboard_data.get("joint_velocities"),
            x=side_x,
            y=margin + header_h + margin,
            w=side_w,
            h=camera_area_h,
        )
        self._draw_popup()
        pygame.display.flip()

    def _draw_header(self, data: Dict[str, Any], x: int, y: int, w: int, h: int) -> None:
        pygame.draw.rect(self._screen, (45, 45, 45), pygame.Rect(x, y, w, h), border_radius=8)

        phase = data.get("phase", "idle")
        traj_idx = data.get("traj_idx", 0)
        total_traj = data.get("total_traj", 0)
        step_idx = data.get("step_idx", 0)
        max_steps = max(1, int(data.get("max_steps", 1)))
        obs_count = int(data.get("obs_count", 0))
        status = data.get("status_text", "")

        header_lines = [
            f"Phase: {phase}",
            f"Trajectory: {traj_idx}/{total_traj}",
            f"Observations collected: {obs_count}",
            "Controls: [S] start  [A] save  [B] discard",
        ]
        if status:
            header_lines.append(f"Status: {status}")

        for i, line in enumerate(header_lines):
            text_surf = self._font.render(line, True, (230, 230, 230))
            self._screen.blit(text_surf, (x + 14, y + 10 + i * 22))

        bar_x = x + int(w * 0.45)
        bar_y = y + h - 38
        bar_w = int(w * 0.5)
        bar_h = 18
        progress = float(np.clip(step_idx / max_steps, 0.0, 1.0))
        pygame.draw.rect(self._screen, (70, 70, 70), pygame.Rect(bar_x, bar_y, bar_w, bar_h), border_radius=6)
        pygame.draw.rect(
            self._screen,
            (60, 180, 75),
            pygame.Rect(bar_x, bar_y, int(bar_w * progress), bar_h),
            border_radius=6,
        )
        prog_text = self._small_font.render(
            f"Episode progress (tqdm): {step_idx}/{max_steps}",
            True,
            (220, 220, 220),
        )
        self._screen.blit(prog_text, (bar_x, bar_y - 22))

    def _draw_camera_grid(self, cameras: Dict[str, np.ndarray], x: int, y: int, w: int, h: int) -> None:
        if not cameras:
            self._draw_text_box("No camera frames available", x, y, w, h)
            return

        names = list(cameras.keys())
        cols = 2 if len(names) > 1 else 1
        rows = int(np.ceil(len(names) / cols))
        tile_w = max(120, (w - (cols + 1) * 8) // cols)
        tile_h = max(100, (h - (rows + 1) * 8) // rows)

        for i, name in enumerate(names):
            row = i // cols
            col = i % cols
            tile_x = x + 8 + col * (tile_w + 8)
            tile_y = y + 8 + row * (tile_h + 8)
            frame = cameras[name]
            self._draw_camera_tile(name, frame, tile_x, tile_y, tile_w, tile_h)

    def _draw_camera_tile(self, name: str, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        pygame.draw.rect(self._screen, (55, 55, 55), pygame.Rect(x, y, w, h), border_radius=6)
        title = self._small_font.render(name, True, (240, 240, 240))
        self._screen.blit(title, (x + 8, y + 6))

        if frame is None:
            self._draw_text_box("Frame unavailable", x + 4, y + 26, w - 8, h - 30)
            return

        frame_np = np.asarray(frame)
        if frame_np.ndim != 3 or frame_np.shape[2] < 3:
            self._draw_text_box("Invalid frame shape", x + 4, y + 26, w - 8, h - 30)
            return

        frame_rgb = frame_np[:, :, :3]
        if frame_rgb.dtype != np.uint8:
            frame_rgb = np.clip(frame_rgb, 0, 255).astype(np.uint8)

        # Resize with OpenCV (fast, releases the GIL) and hand pygame a ready
        # buffer -- no transpose copy, no smoothscale on a full-size surface.
        tw, th = max(1, w - 8), max(1, h - 34)
        small = cv2.resize(np.ascontiguousarray(frame_rgb), (tw, th), interpolation=cv2.INTER_AREA)
        surface = pygame.image.frombuffer(small.tobytes(), (tw, th), "RGB")
        self._screen.blit(surface, (x + 4, y + 30))

    def _draw_joint_panel(
        self,
        joint_positions: Optional[np.ndarray],
        joint_velocities: Optional[np.ndarray],
        x: int,
        y: int,
        w: int,
        h: int,
    ) -> None:
        pygame.draw.rect(self._screen, (40, 40, 40), pygame.Rect(x, y, w, h), border_radius=8)
        title = self._font.render("Joint Telemetry", True, (235, 235, 235))
        self._screen.blit(title, (x + 10, y + 8))

        if joint_positions is None:
            self._draw_text_box("No joint_positions in observations", x + 8, y + 36, w - 16, h - 44)
            return

        q = np.asarray(joint_positions).flatten()
        qd = np.asarray(joint_velocities).flatten() if joint_velocities is not None else None
        line_y = y + 40
        line_h = 20
        max_lines = max(1, (h - 52) // line_h)

        # Left arm joints: indices 0..6
        left_title = self._small_font.render("Left arm (j00-j06)", True, (180, 220, 255))
        self._screen.blit(left_title, (x + 10, line_y))
        line_y += line_h
        for idx in range(0, min(7, len(q))):
            if (line_y - y) // line_h >= max_lines:
                break
            if qd is not None and idx < len(qd):
                text = f"j{idx:02d}: pos={q[idx]: .4f}  vel={qd[idx]: .4f}"
            else:
                text = f"j{idx:02d}: pos={q[idx]: .4f}"
            txt = self._small_font.render(text, True, (220, 220, 220))
            self._screen.blit(txt, (x + 10, line_y))
            line_y += line_h

        line_y += 4
        # Right arm joints: indices 7..13
        if len(q) > 7:
            right_title = self._small_font.render("Right arm (j07-j13)", True, (255, 210, 170))
            self._screen.blit(right_title, (x + 10, line_y))
            line_y += line_h
            for idx in range(7, min(14, len(q))):
                if (line_y - y) // line_h >= max_lines:
                    break
                if qd is not None and idx < len(qd):
                    text = f"j{idx:02d}: pos={q[idx]: .4f}  vel={qd[idx]: .4f}"
                else:
                    text = f"j{idx:02d}: pos={q[idx]: .4f}"
                txt = self._small_font.render(text, True, (220, 220, 220))
                self._screen.blit(txt, (x + 10, line_y))
                line_y += line_h

    def _draw_text_box(self, text: str, x: int, y: int, w: int, h: int) -> None:
        pygame.draw.rect(self._screen, (58, 58, 58), pygame.Rect(x, y, w, h), border_radius=6)
        txt = self._small_font.render(text, True, (220, 220, 220))
        self._screen.blit(txt, (x + 10, y + 10))

    def _show_popup(self, text: str, duration_s: float = 1.2) -> None:
        self._popup_text = text
        self._popup_until = pygame.time.get_ticks() / 1000.0 + duration_s

    def _draw_popup(self) -> None:
        now = pygame.time.get_ticks() / 1000.0
        if now > self._popup_until or not self._popup_text:
            return
        popup_w = 360
        popup_h = 70
        x = (self._screen_width - popup_w) // 2
        y = (self._screen_height - popup_h) // 2
        pygame.draw.rect(self._screen, (20, 20, 20), pygame.Rect(x, y, popup_w, popup_h), border_radius=8)
        pygame.draw.rect(self._screen, (210, 210, 210), pygame.Rect(x, y, popup_w, popup_h), width=2, border_radius=8)
        txt = self._font.render(self._popup_text, True, (245, 245, 245))
        text_rect = txt.get_rect(center=(x + popup_w // 2, y + popup_h // 2))
        self._screen.blit(txt, text_rect)


def main():
    kb = KBReset()
    while True:
        state = kb.update()
        if state == "start":
            print("start")


if __name__ == "__main__":
    main()
