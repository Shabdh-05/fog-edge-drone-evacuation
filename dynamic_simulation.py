import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import time
from datetime import datetime

from intelligence.panic_predictor import PanicPredictor
from intelligence.density_analyzer import DensityAnalyzer
from intelligence.swarm_analyzer import SwarmAnalyzer
from navigation.astar_planner import AStarPlanner
from visualization.draw_heatmap import HeatmapVisualizer
from visualization.draw_corridor import CorridorVisualizer
from fog.regional_load_balancer import RegionalLoadBalancer


# ============================================================
# FOG-EDGE DISASTER RESPONSE - PROFESSIONAL COMMAND DASHBOARD
# ============================================================
# Full-screen, adaptive OpenCV UI.
#
# Existing project modules are preserved:
#   YOLO -> edge detection
#   PanicPredictor -> movement / panic
#   DensityAnalyzer -> crowd density
#   SwarmAnalyzer -> regional risk
#   AStarPlanner -> evacuation route
#   RegionalLoadBalancer -> safe-zone assignment
#
# Controls:
#   Q / ESC : exit
#   F       : toggle fullscreen
#   SPACE   : pause / resume
# ============================================================


# ================= CONFIG =================
GRID_SIZE = 20
YOLO_CONF = 0.25

INITIAL_HAZARD = 80
HAZARD_GROWTH = 0.6
MAX_HAZARD = 260
HAZARD_PANIC_BOOST = 0.8

FUTURE_SCALE = 12
VIDEO_PATH = "assets/disaster.mp4"

# Original safe-zone layout is based on approximately 1000 x 600 video.
SAFE_ZONE_REFERENCE_W = 1000
SAFE_ZONE_REFERENCE_H = 600

# Dashboard tuning
UI_REFERENCE_W = 1920
UI_REFERENCE_H = 1080
PANEL_W = 430

# Colors are BGR because OpenCV uses BGR.
C_BG = (8, 10, 14)
C_PANEL = (10, 13, 18)
C_PANEL_2 = (15, 19, 25)
C_WHITE = (235, 240, 245)
C_MUTED = (145, 155, 165)
C_GREEN = (40, 230, 90)
C_GREEN_DARK = (15, 90, 45)
C_CYAN = (255, 210, 55)
C_BLUE = (255, 125, 40)
C_YELLOW = (40, 215, 255)
C_RED = (40, 45, 245)
C_ORANGE = (40, 150, 255)
C_LINE = (55, 65, 78)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_DUPLEX


# ================= UTILITY FUNCTIONS =================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def put_text(img, text, org, scale=0.6, color=C_WHITE, thickness=1,
             font=FONT, line_type=cv2.LINE_AA):
    cv2.putText(img, str(text), org, font, scale, color, thickness, line_type)


def rounded_rect(img, x1, y1, x2, y2, color, radius=16, thickness=-1):
    """OpenCV rounded rectangle helper."""
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    radius = int(max(0, min(radius, (x2 - x1) // 2, (y2 - y1) // 2)))

    if thickness < 0:
        cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, -1)
        cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1)
        cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1)
        cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1)
    else:
        cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
        cv2.ellipse(img, (x1 + radius, y1 + radius),
                    (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y1 + radius),
                    (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1 + radius, y2 - radius),
                    (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2 - radius, y2 - radius),
                    (radius, radius), 0, 0, 90, color, thickness)


def alpha_rect(img, x1, y1, x2, y2, color, alpha=0.82, radius=16):
    """Draw a translucent rounded panel."""
    overlay = img.copy()
    rounded_rect(overlay, x1, y1, x2, y2, color, radius, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def line_with_glow(img, p1, p2, color, thickness=2):
    """Subtle glow effect for routes / telemetry lines."""
    overlay = img.copy()
    cv2.line(overlay, p1, p2, color, thickness * 4, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.22, img, 0.78, 0, img)
    cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_glow_circle(img, center, radius, color, alpha=0.35):
    overlay = img.copy()
    cv2.circle(overlay, center, radius + 7, color, 2, cv2.LINE_AA)
    cv2.circle(overlay, center, radius + 13, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.circle(img, center, radius, color, -1, cv2.LINE_AA)


def draw_progress_bar(img, x, y, width, height, value, max_value,
                     color=C_GREEN, label=None, value_text=None):
    ratio = 0 if max_value <= 0 else clamp(value / max_value, 0, 1)
    rounded_rect(img, x, y, x + width, y + height, (27, 32, 40), 8, -1)
    if ratio > 0:
        rounded_rect(img, x, y, x + int(width * ratio), y + height,
                     color, 8, -1)
    if label is not None:
        put_text(img, label, (x, y - 8), 0.48, C_MUTED, 1)
    if value_text is not None:
        tw, _ = cv2.getTextSize(str(value_text), FONT, 0.48, 1)[0]
        put_text(img, value_text, (x + width - tw, y - 8), 0.48,
                 C_WHITE, 1)


def draw_section_title(img, x, y, title, accent=C_GREEN):
    cv2.line(img, (x, y + 6), (x + 18, y + 6), accent, 3)
    put_text(img, title.upper(), (x + 28, y + 11), 0.48, C_WHITE, 1)


def draw_metric_card(img, x, y, w, h, title, value, subtitle,
                     accent=C_GREEN, icon=None):
    alpha_rect(img, x, y, x + w, y + h, C_PANEL_2, 0.88, 13)
    cv2.line(img, (x, y + 1), (x, y + h - 1), accent, 3)

    if icon:
        put_text(img, icon, (x + 15, y + 27), 0.48, accent, 1, FONT_MONO)

    put_text(img, title.upper(), (x + 42 if icon else x + 15, y + 22),
             0.40, C_MUTED, 1)
    put_text(img, value, (x + 15, y + 53), 0.88, C_WHITE, 2, FONT_MONO)
    put_text(img, subtitle, (x + 15, y + h - 13), 0.38, C_MUTED, 1)


def fit_video_to_screen(frame, target_w, target_h):
    """
    Preserve the entire camera frame while filling the available screen.
    Letterboxing is used only when the aspect ratios differ.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (nw, nh), interpolation=interp)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x = (target_w - nw) // 2
    y = (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized

    return canvas, x, y, scale


def get_screen_size():
    """Windows-first screen-size detection with safe fallback."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return UI_REFERENCE_W, UI_REFERENCE_H


def create_obstacle_grid(frame, hazard_center, hazard_radius):
    h, w = frame.shape[:2]
    gw = max(1, w // GRID_SIZE)
    gh = max(1, h // GRID_SIZE)

    small = cv2.resize(frame, (gw, gh))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    grid = np.zeros_like(gray)
    grid[gray < 60] = 1

    hx = int(hazard_center[0] // GRID_SIZE)
    hy = int(hazard_center[1] // GRID_SIZE)
    r = max(1, int(hazard_radius // GRID_SIZE))

    yy, xx = np.ogrid[:gh, :gw]
    mask = (xx - hx) ** 2 + (yy - hy) ** 2 < r * r
    grid[mask] = 1

    return grid


def draw_route_glow(frame, scaled_path):
    """Additional polished route rendering around the existing corridor."""
    if not scaled_path or len(scaled_path) < 2:
        return frame

    pts = np.array(scaled_path, dtype=np.int32)

    glow = frame.copy()
    cv2.polylines(glow, [pts], False, C_GREEN, 12, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.16, frame, 0.84, 0, frame)

    cv2.polylines(frame, [pts], False, (80, 255, 130), 3, cv2.LINE_AA)

    # Direction markers
    step = max(1, len(scaled_path) // 7)
    for i in range(step, len(scaled_path), step):
        px, py = scaled_path[i]
        cv2.circle(frame, (int(px), int(py)), 3, C_WHITE, -1, cv2.LINE_AA)

    return frame


def draw_risk_ring(img, center, radius, risk):
    """Professional threat ring."""
    risk = clamp(float(risk), 0, 1)

    cv2.circle(img, center, radius, (40, 40, 45), 2, cv2.LINE_AA)

    cv2.ellipse(img, center, (radius, radius), -90,
                0, int(360 * risk), C_RED, 6, cv2.LINE_AA)

    cv2.ellipse(img, center, (radius, radius), -90,
                int(360 * risk), 360, C_GREEN, 6, cv2.LINE_AA)

    put_text(img, f"{int(risk * 100)}%", (center[0] - 28, center[1] + 8),
             0.72, C_WHITE, 2, FONT_MONO)


def draw_mini_trend(img, x, y, w, h, risk_history, density_history):
    alpha_rect(img, x, y, x + w, y + h, C_PANEL_2, 0.90, 14)
    draw_section_title(img, x + 15, y + 14, "Threat / Density Trend", C_CYAN)

    chart_x = x + 15
    chart_y = y + 43
    chart_w = w - 30
    chart_h = h - 62

    cv2.rectangle(img, (chart_x, chart_y),
                  (chart_x + chart_w, chart_y + chart_h),
                  (30, 36, 44), 1)

    for frac in (0.25, 0.5, 0.75):
        gy = chart_y + int(chart_h * frac)
        cv2.line(img, (chart_x, gy), (chart_x + chart_w, gy),
                 (25, 31, 38), 1)

    def draw_series(history, color):
        if len(history) < 2:
            return
        n = min(len(history), 200)
        vals = list(history)[-n:]
        pts = []
        for i, val in enumerate(vals):
            px = chart_x + int(i / max(1, n - 1) * chart_w)
            py = chart_y + chart_h - int(clamp(float(val), 0, 1) * chart_h)
            pts.append((px, py))
        cv2.polylines(img, [np.array(pts, dtype=np.int32)],
                      False, color, 2, cv2.LINE_AA)

    draw_series(density_history, C_GREEN)
    draw_series(risk_history, C_RED)

    put_text(img, "DENSITY", (chart_x + 8, chart_y + 18),
             0.32, C_GREEN, 1)
    put_text(img, "RISK", (chart_x + 76, chart_y + 18),
             0.32, C_RED, 1)


def draw_safe_zone_card(img, x, y, w, h, zone_id, load, capacity,
                        active=False):
    ratio = clamp(load / max(1, capacity), 0, 1)

    border = C_GREEN if active else C_LINE
    alpha_rect(img, x, y, x + w, y + h, C_PANEL_2, 0.90, 12)
    rounded_rect(img, x, y, x + w, y + h, border, 12, 1)

    status = "ACTIVE ROUTE" if active else "AVAILABLE"
    status_color = C_GREEN if active else C_MUTED

    put_text(img, f"ZONE {zone_id}", (x + 13, y + 20),
             0.38, C_WHITE, 1, FONT_MONO)
    put_text(img, status, (x + 13, y + 36),
             0.27, status_color, 1)

    put_text(img, f"{load}/{capacity}", (x + w - 66, y + 24),
             0.48, C_WHITE, 2, FONT_MONO)

    draw_progress_bar(img, x + 13, y + h - 11, w - 26, 6,
                      load, capacity,
                      C_GREEN if ratio < 0.8 else C_ORANGE)


def draw_top_header(canvas, frame_no, fps, paused):
    h, w = canvas.shape[:2]

    alpha_rect(canvas, 0, 0, w, 76, (5, 8, 12), 0.93, 0)

    cv2.line(canvas, (0, 75), (w, 75), (40, 55, 65), 1)

    cv2.circle(canvas, (28, 38), 9, C_GREEN, -1, cv2.LINE_AA)
    cv2.circle(canvas, (28, 38), 15, C_GREEN, 1, cv2.LINE_AA)

    put_text(canvas, "FOG-EDGE", (54, 34), 0.72, C_WHITE, 2, FONT_MONO)
    put_text(canvas, "DISASTER RESPONSE COMMAND", (55, 56),
             0.34, C_MUTED, 1)

    title = "AUTONOMOUS EVACUATION GUIDANCE"
    tw = cv2.getTextSize(title, FONT_MONO, 0.48, 1)[0][0]
    put_text(canvas, title, ((w - tw) // 2, 38), 0.48, C_CYAN, 1, FONT_MONO)

    status_text = "PAUSED" if paused else "SYSTEM ONLINE"
    status_color = C_ORANGE if paused else C_GREEN

    put_text(canvas, status_text, (w - 250, 30), 0.40,
             status_color, 1, FONT_MONO)
    put_text(canvas, f"FPS {fps:04.1f}", (w - 135, 30),
             0.40, C_WHITE, 1, FONT_MONO)
    put_text(canvas, f"FRAME {frame_no:06d}", (w - 250, 53),
             0.34, C_MUTED, 1, FONT_MONO)

    now = datetime.now().strftime("%H:%M:%S")
    put_text(canvas, now, (w - 105, 53), 0.34, C_MUTED, 1, FONT_MONO)


def draw_video_overlay(frame, camera_w, camera_h, detected, in_hazard,
                       hazard_radius, current_zone, fps):
    """Minimal tactical overlay on the live camera feed."""
    alpha_rect(frame, 18, 92, 250, 132, (5, 8, 12), 0.72, 10)
    put_text(frame, "EDGE CAMERA 01", (32, 113), 0.40, C_GREEN, 1, FONT_MONO)
    put_text(frame, "LIVE / OBJECT ANALYTICS", (32, 127),
             0.29, C_MUTED, 1)

    y = camera_h - 105
    alpha_rect(frame, 18, y, 330, camera_h - 18, (5, 8, 12), 0.72, 12)

    put_text(frame, f"PERSONS DETECTED   {detected:02d}",
             (34, y + 25), 0.42, C_WHITE, 1, FONT_MONO)
    put_text(frame, f"THREAT ZONE        {in_hazard:02d}",
             (34, y + 48), 0.42, C_RED if in_hazard else C_GREEN,
             1, FONT_MONO)
    put_text(frame, f"EVAC ROUTE         ZONE {current_zone}",
             (34, y + 71), 0.42, C_GREEN, 1, FONT_MONO)

    put_text(frame, f"{fps:04.1f} FPS", (camera_w - 105, camera_h - 25),
             0.36, C_MUTED, 1, FONT_MONO)


def build_dashboard(canvas, x, y, w, h,
                    detected, people_in_hazard, hazard_radius,
                    current_zone_id, zone_load, zone_capacity,
                    density_score, avg_panic, swarm_risk,
                    risk_history, density_history, fps,
                    path_count):
    """Right-side professional operations dashboard."""

    alpha_rect(canvas, x, y, x + w, y + h, C_BG, 0.94, 0)

    cv2.line(canvas, (x, y + 76), (x + w, y + 76), (40, 55, 65), 1)

    put_text(canvas, "FOG NODE", (x + 22, y + 32),
             0.62, C_WHITE, 2, FONT_MONO)
    put_text(canvas, "REGIONAL COORDINATION", (x + 23, y + 54),
             0.31, C_MUTED, 1)

    cv2.circle(canvas, (x + w - 32, y + 28), 6, C_GREEN, -1)
    put_text(canvas, "ONLINE", (x + w - 94, y + 33),
             0.32, C_GREEN, 1, FONT_MONO)

    margin = 18
    gap = 8
    card_w = (w - margin * 2 - gap) // 2
    card_h = 74
    cy = y + 86

    draw_metric_card(
        canvas, x + margin, cy, card_w, card_h,
        "Detected", f"{detected:02d}", "edge objects",
        C_GREEN, "D"
    )

    draw_metric_card(
        canvas, x + margin + card_w + gap, cy, card_w, card_h,
        "At Risk", f"{people_in_hazard:02d}", "inside hazard",
        C_RED if people_in_hazard else C_GREEN, "!"
    )

    cy += card_h + 8

    draw_metric_card(
        canvas, x + margin, cy, card_w, card_h,
        "Hazard", f"{int(hazard_radius)}", "radius / px",
        C_ORANGE, "H"
    )

    draw_metric_card(
        canvas, x + margin + card_w + gap, cy, card_w, card_h,
        "Risk", f"{int(swarm_risk * 100)}%", "swarm risk",
        C_RED if swarm_risk > 0.65 else C_ORANGE, "R"
    )

    cy += card_h + 12

    alpha_rect(canvas, x + margin, cy, x + w - margin, cy + 62,
               C_PANEL_2, 0.90, 11)

    draw_section_title(canvas, x + margin + 12, cy + 8,
                       "Threat Assessment", C_RED)

    threat_label = (
        "CRITICAL" if swarm_risk >= 0.75 else
        "HIGH" if swarm_risk >= 0.55 else
        "MODERATE" if swarm_risk >= 0.30 else
        "LOW"
    )

    threat_color = (
        C_RED if swarm_risk >= 0.75 else
        C_ORANGE if swarm_risk >= 0.55 else
        C_YELLOW if swarm_risk >= 0.30 else
        C_GREEN
    )

    put_text(canvas, threat_label, (x + w - 125, cy + 25),
             0.44, threat_color, 2, FONT_MONO)

    draw_progress_bar(
        canvas, x + margin + 13, cy + 39, w - 2 * margin - 26, 8,
        swarm_risk, 1.0, threat_color
    )

    cy += 76
    chart_h = 125

    draw_mini_trend(canvas, x + margin, cy,
                    w - 2 * margin, chart_h,
                    risk_history, density_history)

    cy += chart_h + 10
    ring_h = 125

    alpha_rect(canvas, x + margin, cy, x + w - margin, cy + ring_h,
               C_PANEL_2, 0.90, 14)

    draw_section_title(canvas, x + margin + 14, cy + 12,
                       "AI Risk State", C_CYAN)

    ring_center = (x + margin + 72, cy + 76)
    draw_risk_ring(canvas, ring_center, 45, swarm_risk)

    put_text(canvas, "SWARM RISK", (ring_center[0] - 34, cy + 112),
             0.28, C_MUTED, 1, FONT_MONO)

    values_x = x + margin + 155

    put_text(canvas, "DENSITY", (values_x, cy + 50),
             0.30, C_MUTED, 1)
    put_text(canvas, f"{density_score:.2f}", (values_x, cy + 69),
             0.52, C_GREEN, 2, FONT_MONO)

    put_text(canvas, "PANIC INDEX", (values_x, cy + 91),
             0.30, C_MUTED, 1)
    put_text(canvas, f"{avg_panic:.2f}", (values_x, cy + 110),
             0.52, C_ORANGE, 2, FONT_MONO)

    cy += ring_h + 9

    draw_section_title(canvas, x + margin + 2, cy + 5,
                       "Evacuation Capacity", C_GREEN)

    zone_y = cy + 23
    zone_gap = 6
    zone_h = 50

    for zid in sorted(zone_capacity.keys()):
        draw_safe_zone_card(
            canvas,
            x + margin,
            zone_y,
            w - 2 * margin,
            zone_h,
            zid,
            zone_load.get(zid, 0),
            zone_capacity[zid],
            zid == current_zone_id
        )
        zone_y += zone_h + zone_gap

    footer_y = y + h - 64

    alpha_rect(canvas, x + margin, footer_y,
               x + w - margin, y + h - 16,
               (12, 25, 20), 0.90, 11)

    put_text(canvas, "ROUTING ENGINE", (x + margin + 14, footer_y + 22),
             0.31, C_MUTED, 1, FONT_MONO)
    put_text(canvas, "A* | FOG BALANCING | LIVE",
             (x + margin + 14, footer_y + 43),
             0.40, C_GREEN, 1, FONT_MONO)

    route_text = f"{path_count} ACTIVE ROUTE" + ("S" if path_count != 1 else "")
    tw = cv2.getTextSize(route_text, FONT_MONO, 0.34, 1)[0][0]
    put_text(canvas, route_text, (x + w - margin - tw - 10, footer_y + 32),
             0.34, C_WHITE, 1, FONT_MONO)


# ================= MODEL / MODULE INITIALIZATION =================
model = YOLO("yolov8n.pt")

panic_predictor = PanicPredictor()
density_analyzer = DensityAnalyzer()
swarm_analyzer = SwarmAnalyzer()
astar = AStarPlanner()
heatmap_visualizer = HeatmapVisualizer()
corridor_visualizer = CorridorVisualizer()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open video: {VIDEO_PATH}. "
        "Check that assets/disaster.mp4 exists."
    )


# ================= STATE =================
hazard_center = None
hazard_radius = INITIAL_HAZARD

density_history = deque(maxlen=200)
risk_history = deque(maxlen=200)

frame_no = 0
paused = False

fps_value = 0.0
last_fps_time = time.perf_counter()
fps_counter = 0

fog_node = None
safe_zones = None
zone_capacity = {0: 5, 1: 5, 2: 5}
current_safe_zone_id = 0

screen_w, screen_h = get_screen_size()


# ================= MAIN LOOP =================
WINDOW_NAME = "FOG-EDGE DISASTER RESPONSE"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

try:
    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )
except Exception:
    pass


while True:

    if not paused:
        ret, frame = cap.read()

        if not ret:
            # Restart video automatically for a continuous demonstration.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

            if not ret:
                break

        frame_no += 1

        h, w = frame.shape[:2]

        # Initialize / scale safe zones once.
        if safe_zones is None:
            sx = w / SAFE_ZONE_REFERENCE_W
            sy = h / SAFE_ZONE_REFERENCE_H

            safe_zones = {
                0: (int(200 * sx), int(80 * sy)),
                1: (int(800 * sx), int(80 * sy)),
                2: (int(500 * sx), int(500 * sy)),
            }

            fog_node = RegionalLoadBalancer(
                safe_zones,
                zone_capacity
            )

        if hazard_center is None:
            hazard_center = (w // 2, h // 2)

        if hazard_radius < MAX_HAZARD:
            hazard_radius += HAZARD_GROWTH

        edge_view = frame.copy()

        # ---------------- EDGE NODE ----------------
        cv2.circle(
            edge_view,
            hazard_center,
            int(hazard_radius),
            C_RED,
            2,
            cv2.LINE_AA
        )

        put_text(
            edge_view,
            "HAZARD",
            (
                int(hazard_center[0] - 38),
                int(hazard_center[1] - hazard_radius - 12)
            ),
            0.55,
            C_RED,
            2,
            FONT_MONO
        )

        # YOLO inference
        results = model(edge_view, conf=YOLO_CONF, verbose=False)[0]

        persons = []
        positions = []
        people_in_hazard = 0

        for box in results.boxes:

            if int(box.cls[0]) != 0:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            vx, vy = panic_predictor.compute_velocity(
                id(box),
                (cx, cy)
            )

            persons.append([cx, cy, vx, vy])
            positions.append((cx, cy))

            dist = math.sqrt(
                (cx - hazard_center[0]) ** 2 +
                (cy - hazard_center[1]) ** 2
            )

            in_danger = dist < hazard_radius

            if in_danger:
                people_in_hazard += 1

            # Professional bounding box.
            box_color = C_RED if in_danger else C_GREEN

            cv2.rectangle(
                edge_view,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
                cv2.LINE_AA
            )

            # Current position marker.
            cv2.circle(
                edge_view,
                (cx, cy),
                4,
                C_WHITE,
                -1,
                cv2.LINE_AA
            )

            cv2.line(
                edge_view,
                (cx - 8, cy),
                (cx + 8, cy),
                box_color,
                1,
                cv2.LINE_AA
            )

            cv2.line(
                edge_view,
                (cx, cy - 8),
                (cx, cy + 8),
                box_color,
                1,
                cv2.LINE_AA
            )

            # Future position prediction.
            future_x = int(cx + vx * FUTURE_SCALE)
            future_y = int(cy + vy * FUTURE_SCALE)

            if 0 <= future_x < w and 0 <= future_y < h:

                line_with_glow(
                    edge_view,
                    (cx, cy),
                    (future_x, future_y),
                    C_BLUE,
                    1
                )

                cv2.circle(
                    edge_view,
                    (future_x, future_y),
                    5,
                    C_BLUE,
                    -1,
                    cv2.LINE_AA
                )

                put_text(
                    edge_view,
                    "PRED",
                    (future_x + 7, future_y - 5),
                    0.30,
                    C_BLUE,
                    1,
                    FONT_MONO
                )

        detected = len(persons)

        # ---------------- AI ANALYTICS ----------------
        density_score, _ = density_analyzer.compute_density_score(
            positions
        )

        panic_vals = []

        for cx, cy, vx, vy in persons:

            panic = panic_predictor.compute_panic_index(
                (vx, vy),
                density_score
            )

            dist = math.sqrt(
                (cx - hazard_center[0]) ** 2 +
                (cy - hazard_center[1]) ** 2
            )

            if dist < hazard_radius:
                panic += HAZARD_PANIC_BOOST

            panic_vals.append(panic)

        avg_panic = float(np.mean(panic_vals)) if panic_vals else 0.0

        swarm_risk = float(
            clamp(
                swarm_analyzer.compute_swarm_risk(
                    density_score,
                    avg_panic
                ),
                0,
                1
            )
        )

        density_history.append(float(clamp(density_score, 0, 1)))
        risk_history.append(swarm_risk)

        # ---------------- FOG LOAD BALANCING ----------------
        drone_packet = {
            "drone_id": 1,
            "safe_zone_id": current_safe_zone_id,
            "people_count": detected
        }

        updated = fog_node.balance([drone_packet])

        if updated:
            current_safe_zone_id = updated[0]["safe_zone_id"]

        safe_zone = safe_zones[current_safe_zone_id]

        draw_glow_circle(
            edge_view,
            safe_zone,
            14,
            C_GREEN
        )

        put_text(
            edge_view,
            "SAFE ZONE",
            (safe_zone[0] - 52, safe_zone[1] - 22),
            0.55,
            C_GREEN,
            2,
            FONT_MONO
        )

        # ---------------- HEATMAP ----------------
        try:
            edge_view = heatmap_visualizer.apply_heatmap(
                edge_view,
                positions
            )
        except Exception:
            pass

        # Re-draw critical markers after heatmap.
        cv2.circle(
            edge_view,
            hazard_center,
            int(hazard_radius),
            C_RED,
            2,
            cv2.LINE_AA
        )

        draw_glow_circle(
            edge_view,
            safe_zone,
            14,
            C_GREEN
        )

        # ---------------- A* ROUTING ----------------
        grid = create_obstacle_grid(
            edge_view,
            hazard_center,
            hazard_radius
        )

        path_count = 0

        for cx, cy, vx, vy in persons:

            start = (
                int(cy // GRID_SIZE),
                int(cx // GRID_SIZE)
            )

            goal = (
                int(safe_zone[1] // GRID_SIZE),
                int(safe_zone[0] // GRID_SIZE)
            )

            try:
                path = astar.find_path(grid, start, goal)
            except Exception:
                path = None

            if path:

                scaled_path = [
                    (
                        int(p[1] * GRID_SIZE),
                        int(p[0] * GRID_SIZE)
                    )
                    for p in path
                ]

                try:
                    edge_view = corridor_visualizer.draw_corridor(
                        edge_view,
                        scaled_path
                    )
                except Exception:
                    pass

                edge_view = draw_route_glow(
                    edge_view,
                    scaled_path
                )

                path_count += 1

        # ---------------- FPS ----------------
        fps_counter += 1
        now_perf = time.perf_counter()

        if now_perf - last_fps_time >= 0.5:
            fps_value = fps_counter / (now_perf - last_fps_time)
            fps_counter = 0
            last_fps_time = now_perf

        # ---------------- CAMERA HUD ----------------
        draw_video_overlay(
            edge_view,
            w,
            h,
            detected,
            people_in_hazard,
            hazard_radius,
            current_safe_zone_id,
            fps_value
        )

    # ========================================================
    # FULL-SCREEN COMPOSITION
    # ========================================================
    canvas, offset_x, offset_y, video_scale = fit_video_to_screen(
        edge_view,
        screen_w,
        screen_h
    )

    panel_w = int(
        clamp(
            PANEL_W * screen_w / UI_REFERENCE_W,
            360,
            520
        )
    )

    panel_x = screen_w - panel_w

    alpha_rect(
        canvas,
        panel_x,
        0,
        screen_w,
        screen_h,
        C_BG,
        0.93,
        0
    )

    cv2.line(
        canvas,
        (panel_x, 0),
        (panel_x, screen_h),
        (45, 65, 70),
        1
    )

    draw_top_header(
        canvas,
        frame_no,
        fps_value,
        paused
    )

    zone_load = fog_node.compute_zone_load(updated) if fog_node else {
        0: 0, 1: 0, 2: 0
    }

    build_dashboard(
        canvas,
        panel_x,
        76,
        panel_w,
        screen_h - 76,
        detected if "detected" in locals() else 0,
        people_in_hazard if "people_in_hazard" in locals() else 0,
        hazard_radius,
        current_safe_zone_id,
        zone_load,
        zone_capacity,
        density_score if "density_score" in locals() else 0,
        avg_panic if "avg_panic" in locals() else 0,
        swarm_risk if "swarm_risk" in locals() else 0,
        risk_history,
        density_history,
        fps_value,
        path_count if "path_count" in locals() else 0
    )

    cv2.rectangle(
        canvas,
        (0, 76),
        (panel_x - 1, screen_h - 1),
        (35, 45, 52),
        1
    )

    strip_y = screen_h - 38

    alpha_rect(
        canvas,
        0,
        strip_y,
        panel_x,
        screen_h,
        (5, 8, 12),
        0.78,
        0
    )

    put_text(
        canvas,
        "EDGE ANALYTICS",
        (18, strip_y + 24),
        0.34,
        C_GREEN,
        1,
        FONT_MONO
    )

    put_text(
        canvas,
        "YOLO | MOTION PREDICTION | DENSITY | A* EVACUATION",
        (145, strip_y + 24),
        0.31,
        C_MUTED,
        1,
        FONT_MONO
    )

    cv2.imshow(WINDOW_NAME, canvas)

    key = cv2.waitKey(1) & 0xFF

    if key in (ord("q"), ord("Q"), 27):
        break

    if key == ord("f") or key == ord("F"):
        try:
            current = cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_FULLSCREEN
            )

            if current == cv2.WINDOW_FULLSCREEN:
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL
                )
            else:
                cv2.setWindowProperty(
                    WINDOW_NAME,
                    cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_FULLSCREEN
                )
        except Exception:
            pass

    if key == 32:
        paused = not paused


cap.release()
cv2.destroyAllWindows()