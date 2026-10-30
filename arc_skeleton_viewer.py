"""
ARC - Pygame 3D Skeleton Viewer
================================
Standalone viewer that reads from the UDP stream (port 8888).
Run this ALONGSIDE mocap_ue5_final.py

Install: pip install pygame
Run:     python arc_skeleton_viewer.py
"""

import math
import time
from collections import deque

import pygame

from config.settings import UDP_PORT, WIDTH, HEIGHT, TARGET_FPS, TITLE
from streaming.udp_receiver import UDPReceiver
from viewer.constants import (
    C_BG, C_GREEN, C_GREEN_DIM, C_CYAN, C_ORANGE,
    C_WHITE, C_GRAY, C_DARKGRAY, C_PANEL,
    CONNECTIONS, KEY_JOINTS,
)
from viewer.projection import project_3d, lerp_color
from viewer import renderer


def main():
    global WIDTH, HEIGHT

    pygame.init()
    renderer.init_fonts()

    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption(TITLE)
    clock = pygame.time.Clock()

    udp = UDPReceiver(UDP_PORT)

    # Camera state
    cam_angle_y = 0.0
    cam_angle_x = 0.25
    auto_rotate  = True
    show_labels  = True
    show_coords  = True

    fps_history = deque([0] * 60, maxlen=60)
    start_time  = time.time()

    running = True
    while running:
        dt = clock.tick(TARGET_FPS) / 1000.0

        # ── Events ───────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.VIDEORESIZE:
                WIDTH, HEIGHT = event.w, event.h
                screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:
                    auto_rotate = not auto_rotate
                if event.key == pygame.K_SPACE:
                    cam_angle_y = 0.0
                    cam_angle_x = 0.25
                if event.key == pygame.K_l:
                    show_labels = not show_labels
                if event.key == pygame.K_c:
                    show_coords = not show_coords
                if event.key == pygame.K_LEFT:
                    cam_angle_y -= 0.15
                if event.key == pygame.K_RIGHT:
                    cam_angle_y += 0.15
                if event.key == pygame.K_UP:
                    cam_angle_x = max(-0.5, cam_angle_x - 0.1)
                if event.key == pygame.K_DOWN:
                    cam_angle_x = min(1.0, cam_angle_x + 0.1)

        if auto_rotate:
            cam_angle_y += 0.008

        # ── UDP data ─────────────────────────────────────────────────────────
        state     = udp.get_state()
        joints    = state['joints']
        connected = state['connected']
        pps       = state['pps']
        seq       = state['sequence']

        # ── Dynamic layout ────────────────────────────────────────────────────
        SKEL_CX    = WIDTH // 2 + 60
        SKEL_CY    = HEIGHT // 2 + 30
        SKEL_SCALE = (HEIGHT / 720) * 2.2

        screen.fill(C_BG)

        renderer.draw_grid(
            screen, SKEL_CX, SKEL_CY + int(220 * HEIGHT / 720),
            size=int(500 * HEIGHT / 720), step=60,
        )

        # ── Project all joints to 2D ─────────────────────────────────────────
        projected = {}
        for jid, j in joints.items():
            pos_raw = j.get('position', [0.5, 0.5, 0.0])
            up  = j.get('unreal_position', {})
            ux  = float(up['x']) if up and 'x' in up else float(pos_raw[2]) * 200.0
            uy  = float(up['y']) if up and 'y' in up else (float(pos_raw[0]) - 0.5) * 200.0
            uz  = float(up['z']) if up and 'z' in up else -(float(pos_raw[1]) - 0.5) * 200.0
            sx, sy, persp = project_3d(ux, uy, uz, SKEL_CX, SKEL_CY,
                                       SKEL_SCALE, cam_angle_x, cam_angle_y)
            projected[jid] = (sx, sy, persp, j.get('visibility', 1.0),
                              j.get('interpolated', False))

        # ── Draw connections (back to front by depth) ────────────────────────
        conn_depth = []
        for (a, b, col, thick) in CONNECTIONS:
            if a in projected and b in projected:
                ax, ay, ap, _, _ = projected[a]
                bx, by, bp, _, _ = projected[b]
                conn_depth.append(((ap + bp) / 2, ax, ay, bx, by, col, thick))

        conn_depth.sort(key=lambda c: c[0])

        for (depth, ax, ay, bx, by, col, thick) in conn_depth:
            glow = lerp_color((0, 0, 0), col, 0.3)
            pygame.draw.line(screen, glow, (ax, ay), (bx, by), thick + 4)
            pygame.draw.line(screen, col,  (ax, ay), (bx, by), thick)

        # ── Draw joints ──────────────────────────────────────────────────────
        for jid, (sx, sy, persp, vis, interp) in projected.items():
            _, base_col = KEY_JOINTS.get(jid, (None, C_GREEN_DIM))
            source = joints.get(jid, {}).get('source', 'front')

            if source == 'back':
                col = (255, 180, 0)
            elif interp:
                col = C_ORANGE
            elif vis > 0.7:
                col = base_col
            else:
                col = lerp_color(C_DARKGRAY, base_col, vis)

            r = max(3, int(6 * persp))
            pygame.draw.circle(screen, lerp_color(C_BG, col, 0.3), (sx, sy), r + 5)
            pygame.draw.circle(screen, col, (sx, sy), r)
            pygame.draw.circle(screen, col, (sx, sy), r + 2, 1)

        # ── Joint labels ─────────────────────────────────────────────────────
        if show_labels:
            for jid, (sx, sy, persp, vis, interp) in projected.items():
                if jid not in KEY_JOINTS:
                    continue
                label, col = KEY_JOINTS[jid]
                tw, th = renderer.FONT_TINY.size(label)
                lx, ly = sx + 10, sy - 8
                pygame.draw.rect(screen, (0, 0, 0), (lx - 2, ly - 2, tw + 4, th + 4))
                screen.blit(renderer.FONT_TINY.render(label, True, col), (lx, ly))

                if show_coords and jid in joints:
                    j       = joints[jid]
                    pos_raw = j.get('position', [0.5, 0.5, 0.0])
                    up      = j.get('unreal_position', {})
                    ux  = float(up['x']) if up and 'x' in up else float(pos_raw[2]) * 200.0
                    uy  = float(up['y']) if up and 'y' in up else (float(pos_raw[0]) - 0.5) * 200.0
                    uz  = float(up['z']) if up and 'z' in up else -(float(pos_raw[1]) - 0.5) * 200.0
                    coord = f"({ux:+.1f},{uy:+.1f},{uz:+.1f})"
                    cw, ch = renderer.FONT_TINY.size(coord)
                    pygame.draw.rect(screen, (0, 0, 0), (lx - 2, ly + th + 1, cw + 4, ch + 2))
                    screen.blit(renderer.FONT_TINY.render(coord, True, C_GRAY), (lx, ly + th + 2))

        # ── Left panel — system stats ─────────────────────────────────────────
        PANEL_X, PANEL_Y, PANEL_W = 8, 8, 250
        PANEL_H = min(220, HEIGHT - 280)

        content_y = renderer.draw_panel(screen, PANEL_X, PANEL_Y, PANEL_W, PANEL_H,
                                        "// ARC SYSTEM STATUS")

        render_fps = clock.get_fps()
        fps_history.append(render_fps)

        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "RENDER FPS", f"{render_fps:.1f}",
            C_GREEN if render_fps >= 50 else C_ORANGE)
        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "STREAM PPS", f"{pps:.1f}",
            C_GREEN if pps > 25 else C_ORANGE)
        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "CONNECTION", "LIVE ●" if connected else "WAITING ○",
            C_GREEN if connected else C_ORANGE)
        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "SEQUENCE", f"#{seq:06d}", C_WHITE)
        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "JOINTS", f"{len(joints)} / 33",
            C_GREEN if len(joints) >= 30 else C_ORANGE)
        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "CAM ANGLE Y", f"{math.degrees(cam_angle_y % (2*math.pi)):.1f}°", C_GRAY)
        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "AUTO ROTATE", "ON" if auto_rotate else "OFF",
            C_GREEN if auto_rotate else C_GRAY)
        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "LABELS", "ON" if show_labels else "OFF",
            C_GREEN if show_labels else C_GRAY)
        content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
            "UPTIME", f"{int(time.time()-start_time)}s", C_GRAY)

        back_joints  = sum(1 for j in joints.values() if j.get('source') == 'back')
        fused_joints = sum(1 for j in joints.values() if j.get('fused'))
        if back_joints > 0 or fused_joints > 0:
            content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
                "BACK CAM", f"{back_joints}/33 joints", (255, 180, 0))
            content_y = renderer.draw_stat_row(screen, PANEL_X+10, content_y,
                "BLENDED", f"{fused_joints}/33 joints", C_CYAN)

        # Mini FPS graph
        content_y += 4
        pygame.draw.rect(screen, C_DARKGRAY, (PANEL_X+10, content_y, PANEL_W-20, 30))
        for i, f in enumerate(fps_history):
            bh  = int((f / 70) * 28)
            bx  = PANEL_X + 10 + i * ((PANEL_W-20) // 60)
            col = C_GREEN if f >= 50 else C_ORANGE
            pygame.draw.line(screen, col, (bx, content_y+28), (bx, content_y+28-bh), 2)
        screen.blit(renderer.FONT_TINY.render("FPS HISTORY", True, C_GREEN_DIM),
                    (PANEL_X+10, content_y+2))

        # Innovation badges
        badge_y = PANEL_Y + 230
        bx = PANEL_X + 10
        bx = renderer.draw_badge(screen, bx, badge_y, "INN#1 SMOOTH",    (0, 130, 80))
        bx = renderer.draw_badge(screen, bx, badge_y, "INN#2 OCCL",      (0, 100, 160))
        renderer.draw_badge(screen, PANEL_X+10, badge_y+22, "INN#3 FABRIK-IK", (160, 100, 0))

        # ── Right panel — joint coordinates ──────────────────────────────────
        RP_X, RP_Y, RP_W = WIDTH - 258, 8, 250
        RP_H = min(420, HEIGHT - 50)

        content_y = renderer.draw_panel(screen, RP_X, RP_Y, RP_W, RP_H,
                                        "// JOINT COORDINATES (UE cm)")

        for jid in [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]:
            if jid not in joints:
                continue
            j       = joints[jid]
            pos_raw = j.get('position', [0.5, 0.5, 0.0])
            up      = j.get('unreal_position', {})
            ux  = float(up['x']) if up and 'x' in up else float(pos_raw[2]) * 200.0
            uy  = float(up['y']) if up and 'y' in up else (float(pos_raw[0]) - 0.5) * 200.0
            uz  = float(up['z']) if up and 'z' in up else -(float(pos_raw[1]) - 0.5) * 200.0
            vis = j.get('visibility', 0)

            name, col = KEY_JOINTS.get(jid, (f"J{jid:02d}", C_GRAY))
            screen.blit(renderer.FONT_TINY.render(f"{name:<10}", True, col),
                        (RP_X + 10, content_y))
            screen.blit(renderer.FONT_TINY.render(
                f"X{ux:+6.1f} Y{uy:+6.1f} Z{uz:+6.1f}", True, C_GRAY),
                (RP_X + 10, content_y + 11))

            bar_w   = int((RP_W - 20) * vis)
            bar_col = col if vis > 0.7 else C_ORANGE
            pygame.draw.rect(screen, C_DARKGRAY, (RP_X+10, content_y+22, RP_W-20, 2))
            pygame.draw.rect(screen, bar_col,    (RP_X+10, content_y+22, bar_w,   2))
            content_y += 30

        # ── Bottom bar ────────────────────────────────────────────────────────
        BB_Y = HEIGHT - 36
        pygame.draw.rect(screen, C_PANEL,    (0, BB_Y, WIDTH, 36))
        pygame.draw.rect(screen, C_GREEN_DIM,(0, BB_Y, WIDTH, 36), 1)
        pygame.draw.rect(screen, C_GREEN,    (0, BB_Y, WIDTH, 2))

        controls = [
            ("[R] Auto-rotate", auto_rotate),
            ("[L] Labels",      show_labels),
            ("[C] Coords",      show_coords),
            ("[SPACE] Reset view", True),
            ("[←→] Pan camera",   True),
            ("[↑↓] Tilt camera",  True),
            ("[ESC] Quit",        True),
        ]
        hx = 16
        for txt, active in controls:
            col = C_GREEN if active else C_GRAY
            hf  = renderer.FONT_TINY.render(txt, True, col)
            screen.blit(hf, (hx, BB_Y + 12))
            hx += hf.get_width() + 24

        legend_items = [
            ((0, 255, 120),  "FRONT HIGH"),
            ((0, 200, 255),  "INTERPOLATED"),
            ((255, 180, 0),  "BACK CAM"),
            ((255, 120, 40), "LOW CONF"),
        ]
        lx = hx + 20
        for lcol, ltxt in legend_items:
            pygame.draw.circle(screen, lcol, (lx, BB_Y + 18), 4)
            lf = renderer.FONT_TINY.render(ltxt, True, lcol)
            screen.blit(lf, (lx + 8, BB_Y + 12))
            lx += lf.get_width() + 22

        title_txt = f"ARC // SEQ #{seq:06d}  |  {pps:.1f} pps  |  UDP:{UDP_PORT}"
        tf = renderer.FONT_SMALL.render(title_txt, True, C_GREEN_DIM)
        screen.blit(tf, (WIDTH - tf.get_width() - 12, BB_Y + 10))

        # ── Title bar ─────────────────────────────────────────────────────────
        pygame.draw.rect(screen, (12, 20, 14), (0, 0, WIDTH, 22))
        pygame.draw.rect(screen, C_GREEN_DIM, (0, 21, WIDTH, 1))
        dual_cam = any(j.get('source') == 'back' for j in joints.values())
        dual_txt = "DUAL CAM FUSION" if dual_cam else "SINGLE CAM"
        header = renderer.FONT_TINY.render(
            f"ARC  //  Adaptive Realtime Capture  |  3D Skeleton Viewer  |  "
            f"{dual_txt}  |  Port {UDP_PORT}  |  {'● LIVE' if connected else '○ WAITING FOR STREAM'}",
            True, C_GREEN if connected else C_ORANGE)
        screen.blit(header, (10, 5))

        # ── Waiting message ───────────────────────────────────────────────────
        if not connected:
            msg1 = renderer.FONT_LARGE.render("WAITING FOR STREAM", True, C_ORANGE)
            msg2 = renderer.FONT_SMALL.render("Run:  python mocap_ue5_final.py", True, C_GRAY)
            msg3 = renderer.FONT_SMALL.render(f"Listening on UDP port {UDP_PORT}...", True, C_GREEN_DIM)
            screen.blit(msg1, (SKEL_CX - msg1.get_width()//2, SKEL_CY - 40))
            screen.blit(msg2, (SKEL_CX - msg2.get_width()//2, SKEL_CY + 10))
            screen.blit(msg3, (SKEL_CX - msg3.get_width()//2, SKEL_CY + 36))

        pygame.display.flip()

    udp.stop()
    pygame.quit()
    print("[ARC] Viewer closed.")


if __name__ == "__main__":
    print("=" * 60)
    print("  ARC  //  3D Skeleton Viewer")
    print("=" * 60)
    print(f"  Listening on UDP port {UDP_PORT}")
    print(f"  Make sure mocap_ue5_final.py is running")
    print()
    print("  Controls:")
    print("    R         Toggle auto-rotate")
    print("    L         Toggle joint labels")
    print("    C         Toggle coordinates")
    print("    ← →       Pan camera")
    print("    ↑ ↓       Tilt camera")
    print("    ESC       Quit")
    print("=" * 60)
    main()
