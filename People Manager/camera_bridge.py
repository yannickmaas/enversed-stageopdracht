import argparse
import asyncio
import json
import sys
import threading
import time

import cv2
import numpy as np
import websockets

# ============================================================
# Configuration
# ============================================================
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
MIRROR_X = False          # True if left/right is swapped relative to the table
FLIP_Y = False            # True if top/bottom is swapped (camera mounted upside down)
SHOW_PREVIEW = True       # camera window + mask window for debugging

HOST = "127.0.0.1"
PORT = 8765
SEND_HZ = 30              # maximum number of messages per second to the browser

# Original detection values for the white test object.
BRIGHTNESS_MIN = 200
SATURATION_MAX = 145
MIN_AREA = 3000
MAX_AREA = 150000
ASPECT_MIN, ASPECT_MAX = 0.65, 3.0
EXTENT_MIN = 0.30
SOLIDITY_MIN = 0.70

# Contours touching the edge of the camera frame are ignored.
# Cause of the "cursor stuck in a corner": a bright window, lamp, light wall or
# overexposed table edge at the frame border also passes the white filters and
# then wins as the largest candidate. An object on the table is fully in view.
REJECT_BORDER_TOUCHING = True
BORDER_MARGIN = 4         # pixels

# Detection profiles: one profile per color, each with one or more HSV ranges
# (OpenCV HSV: H 0..180, S 0..255, V 0..255).
# Phase 1 only uses the white test object. Colored LEGO profiles must be
# calibrated under your own lighting; the values below are only a starting
# point and are therefore disabled.
PROFILES = [
    {
        "color": "white",
        "ranges": [((0, 0, BRIGHTNESS_MIN), (180, SATURATION_MAX, 255))],
        "max_objects": 1,
    },
    # {"color": "red",    "ranges": [((0, 120, 70), (10, 255, 255)),
    #                                ((170, 120, 70), (180, 255, 255))], "max_objects": 4},
    # {"color": "blue",   "ranges": [((100, 120, 60), (130, 255, 255))], "max_objects": 4},
    # {"color": "yellow", "ranges": [((20, 120, 120), (35, 255, 255))], "max_objects": 4},
]

# Table -> canvas calibration (preparation for phase 2).
# None = the full camera frame is the play area.
# Otherwise four corner points in camera pixels, in this order:
# top-left, top-right, bottom-right, bottom-left (as on the canvas).
# Tip: click in the camera window; the pixel coordinates are printed in the terminal.
TABLE_CORNERS = None
# TABLE_CORNERS = [(212, 95), (1080, 102), (1105, 650), (190, 640)]

# ------------------------------------------------------------
# Physical chips (fiches)
# ------------------------------------------------------------
# Red chip with a yellow segment = "redirect": the yellow segment points in the new
# walking direction. Green chip = "split": splits the flow into two branches.
# HSV ranges measured on photos of the actual chips (red H 2-8, yellow H 19-28,
# green H 69-82) with margin. Click in the camera window to print the HSV value of
# a pixel and fine-tune these ranges for your own lighting.
CHIP_DETECTION = True
RED_RANGES = [((0, 120, 130), (12, 255, 255)), ((170, 120, 130), (180, 255, 255))]
YELLOW_RANGES = [((17, 120, 150), (36, 255, 255))]
GREEN_RANGES = [((60, 100, 90), (86, 255, 255))]

CHIP_MIN_RADIUS = 12       # px in the camera frame; print with --debug to tune
CHIP_MAX_RADIUS = 110      # px
CHIP_FILL_MIN = 0.60       # share of colored pixels inside the chip circle (rejects rings and outlines)
CHIP_SURROUND_MAX = 0.35   # max. share of the same color in the ring just outside the chip (rejects bars and areas)
RED_MIN_FRACTION = 0.35    # red share of a red chip
YELLOW_FRACTION = (0.04, 0.50)  # the yellow segment must be present, but cannot be the whole chip
CHIP_CLOSE_PX = 9          # closes gaps caused by reflections, tape and printed text

# How many chips of each kind may be active at the same time (largest first).
CHIP_KINDS = {
    "redirect": {"color": "red", "max_objects": 1},
    "split": {"color": "green", "max_objects": 1},
}

# Stable recognition: a chip must be seen in a few consecutive frames before it
# counts, and stays active for a short while when it briefly disappears
# (hand over the chip, reflection), so it does not flicker in the game.
TRACK_CONFIRM_FRAMES = 3
TRACK_HOLD_S = 0.6           # seconds a confirmed chip survives without detection
TRACK_POS_SMOOTH_S = 0.05     # time constant of the position smoothing (seconds; lower = follows faster)
TRACK_HEADING_SMOOTH_S = 0.08 # time constant of the heading smoothing (seconds)

WINDOW_CAMERA = "People Manager - Camera"
WINDOW_MASK = "Detection Mask"


# ============================================================
# Camera capture in a dedicated thread
# ============================================================
class CameraReader:
    """Opens and reads the camera in one fixed thread.

    This keeps the asyncio loop free for the WebSocket, and processing always
    uses the newest frame (older frames are skipped, which avoids lag).
    """

    def __init__(self, index, width, height):
        self.index, self.width, self.height = index, width, height
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.error = None
        self.backend = None
        self.size = None
        self._thread = threading.Thread(target=self._run, name="camera", daemon=True)

    def start(self, timeout=10.0):
        self._thread.start()
        if not self._ready.wait(timeout):
            self.error = "Camera reageert niet binnen {:.0f} s.".format(timeout)
        return self.error is None

    def _open(self):
        attempts = []
        if sys.platform.startswith("win"):
            attempts.append(("DirectShow", cv2.CAP_DSHOW))
        attempts.append(("standaard", cv2.CAP_ANY))
        for name, api in attempts:
            cap = cv2.VideoCapture(self.index, api)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                ok, frame = cap.read()
                if ok and frame is not None:
                    self.backend = name
                    self.size = (frame.shape[1], frame.shape[0])
                    return cap, frame
            cap.release()
        return None, None

    def _run(self):
        cap, frame = self._open()
        if cap is None:
            self.error = (
                f"Camera {self.index} niet gevonden of in gebruik.\n"
                "  - Sluit apps die de camera gebruiken (Teams, Zoom, Camera-app, browsertab).\n"
                "  - Probeer een andere index: --camera 1\n"
                "  - Controleer Windows-instellingen > Privacy > Camera."
            )
            self._ready.set()
            return
        self._publish(frame)
        self._ready.set()

        failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= 50:  # ~a few seconds without an image
                        self.error = "Camerabeeld weggevallen (camera losgekoppeld?)."
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                self._publish(frame)
        finally:
            cap.release()
            with self._cond:
                self._cond.notify_all()

    def _publish(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._cond.notify_all()

    def wait_frame(self, last_seq, timeout=1.0):
        """Blocks until a frame newer than last_seq is available."""
        with self._cond:
            self._cond.wait_for(
                lambda: self._seq != last_seq or self.error or self._stop.is_set(),
                timeout,
            )
            return self._seq, self._frame

    @property
    def alive(self):
        return self._thread.is_alive() and self.error is None

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=2.0)


# ============================================================
# Detection (original white detection, extended to profiles)
# ============================================================
KERNEL_OPEN = np.ones((3, 3), np.uint8)
KERNEL_CLOSE = np.ones((5, 5), np.uint8)


def profile_mask(hsv, profile):
    mask = None
    for lower, upper in profile["ranges"]:
        m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL_CLOSE)
    return mask


def find_candidates(mask, rejected=None):
    """Same contour filters as the original detect_case(),
    plus a check against the frame border."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_h, mask_w = mask.shape[:2]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not MIN_AREA <= area <= MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w == 0 or h == 0:
            continue
        aspect_ratio = w / float(h)
        extent = area / float(w * h)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        if not ASPECT_MIN <= aspect_ratio <= ASPECT_MAX:
            continue
        if extent < EXTENT_MIN or solidity < SOLIDITY_MIN:
            continue
        if REJECT_BORDER_TOUCHING and (
            x <= BORDER_MARGIN or y <= BORDER_MARGIN
            or x + w >= mask_w - BORDER_MARGIN or y + h >= mask_h - BORDER_MARGIN
        ):
            if rejected is not None:
                rejected.append({"box": (x, y, w, h), "reason": "raakt rand"})
            continue

        # Centroid: more accurate than the bounding-box center
        # once a block lies at an angle.
        mom = cv2.moments(contour)
        if mom["m00"]:
            cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        else:
            cx, cy = x + w / 2.0, y + h / 2.0

        # Orientation (preparation for phase 2). For a rectangle this is at best
        # reliable within 0..90 degrees; for a square block it is ambiguous.
        (_, _), (rw, rh), rangle = cv2.minAreaRect(contour)
        if rw < rh:
            rangle += 90.0

        candidates.append({
            "box": (x, y, w, h),
            "area": area,
            "center": (cx, cy),
            "angle": rangle % 180.0,
            "contour": contour,
        })
    return candidates


def detect_objects(frame, rejected=None):
    """Returns (objects, combined mask), largest first per profile."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    combined = None
    objects = []
    for profile in PROFILES:
        mask = profile_mask(hsv, profile)
        combined = mask if combined is None else cv2.bitwise_or(combined, mask)
        found = sorted(find_candidates(mask, rejected), key=lambda c: c["area"], reverse=True)
        for c in found[: profile.get("max_objects", 1)]:
            c["color"] = profile["color"]
            objects.append(c)
    return objects, combined


# ============================================================
# Chip detection (red chip with yellow direction segment, green chip)
# ============================================================
def ranges_mask(hsv, ranges):
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def _clean(mask, close_px):
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL_OPEN)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def circle_candidates(mask, colored, rejected=None, label=""):
    """Round chips in `mask` (cleaned) via the largest inscribed circle.

    A chip often touches other things of the same color: a person in a red shirt
    walking past, a digital red triangle, the yellow segment sticking out. The
    outline of such a merged blob is not round, but the largest circle that fits
    inside it is still the chip. A candidate counts as a chip when that circle is
    well filled with color (`colored` = raw mask) and the ring just outside it is
    mostly empty, so bars, banners and large areas are rejected."""
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    h_img, w_img = mask.shape[:2]
    found = []
    for _ in range(6):
        _, r, _, (px, py) = cv2.minMaxLoc(dt)
        if r < CHIP_MIN_RADIUS * 0.8:
            break
        # refine the center: centroid of the top of the distance transform
        x0, x1, y0, y1 = max(0, int(px - r)), min(w_img, int(px + r + 1)), max(0, int(py - r)), min(h_img, int(py + r + 1))
        win = dt[y0:y1, x0:x1]
        ys, xs = np.nonzero(win >= r * 0.9)
        cx, cy = x0 + xs.mean(), y0 + ys.mean()
        cv2.circle(dt, (int(px), int(py)), int(r * 2.2), 0, -1)  # suppress this peak, look for the next one
        box = (int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        # evaluate only a small window around the candidate (fast)
        R = int(1.7 * r) + 2
        wx0, wx1, wy0, wy1 = max(0, int(cx) - R), min(w_img, int(cx) + R + 1), max(0, int(cy) - R), min(h_img, int(cy) + R + 1)
        yy, xx = np.ogrid[wy0:wy1, wx0:wx1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        disc = d2 <= r * r
        ring = (d2 >= (1.3 * r) ** 2) & (d2 <= (1.7 * r) ** 2)
        fill = np.count_nonzero(colored[wy0:wy1, wx0:wx1][disc]) / max(1, np.count_nonzero(disc))
        around = np.count_nonzero(mask[wy0:wy1, wx0:wx1][ring]) / max(1, np.count_nonzero(ring))
        reason = None
        if r > CHIP_MAX_RADIUS:
            reason = "te groot"
        elif fill < CHIP_FILL_MIN:
            reason = "niet gevuld"
        elif around > CHIP_SURROUND_MAX:
            reason = "niet rond (deel van groter vlak)"
        elif REJECT_BORDER_TOUCHING and (cx - r <= BORDER_MARGIN or cy - r <= BORDER_MARGIN or
                                         cx + r >= w_img - BORDER_MARGIN or cy + r >= h_img - BORDER_MARGIN):
            reason = "raakt rand"
        if reason:
            if rejected is not None:
                rejected.append({"box": box, "reason": f"{label}{reason}"})
            continue
        found.append({"center": (float(cx), float(cy)), "r": float(r), "area": float(np.pi * r * r), "box": box})
    return found


def detect_chips(hsv, rejected=None):
    """Returns (chips, masks). Each chip: kind, color, center, r (px), heading
    (degrees in the camera image, redirect only), head (yellow centroid, px)."""
    red_raw = ranges_mask(hsv, RED_RANGES)
    yellow_raw = ranges_mask(hsv, YELLOW_RANGES)
    green_raw = ranges_mask(hsv, GREEN_RANGES)
    chips = []

    # Red chip: red and yellow together form the round chip; the yellow segment
    # covers part of the red disc (and sticks out a little).
    union_raw = cv2.bitwise_or(red_raw, yellow_raw)
    union = _clean(union_raw, CHIP_CLOSE_PX)
    h_img, w_img = hsv.shape[:2]
    for b in circle_candidates(union, union_raw, rejected, "rood: "):
        cx, cy = b["center"]
        R = int(b["r"] * 1.25) + 2  # the yellow segment may stick out of the red disc
        x0, x1, y0, y1 = max(0, int(cx) - R), min(w_img, int(cx) + R + 1), max(0, int(cy) - R), min(h_img, int(cy) + R + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        disc, wide = d2 <= b["r"] ** 2, d2 <= (b["r"] * 1.25) ** 2
        disc_px = max(1, np.count_nonzero(disc))
        red_frac = np.count_nonzero(red_raw[y0:y1, x0:x1][disc]) / disc_px
        yel_local = (yellow_raw[y0:y1, x0:x1] > 0) & wide
        yel_frac = np.count_nonzero(yel_local) / disc_px
        if red_frac < RED_MIN_FRACTION or not YELLOW_FRACTION[0] <= yel_frac <= YELLOW_FRACTION[1]:
            if rejected is not None:
                why = "te weinig rood" if red_frac < RED_MIN_FRACTION else "geen geel segment" if yel_frac < YELLOW_FRACTION[0] else "te veel geel"
                rejected.append({"box": b["box"], "reason": f"rood: {why}"})
            continue
        ys, xs = np.nonzero(yel_local)
        hx, hy = x0 + xs.mean(), y0 + ys.mean()
        if np.hypot(hx - cx, hy - cy) < b["r"] * 0.15:
            if rejected is not None:
                rejected.append({"box": b["box"], "reason": "rood: richting onduidelijk"})
            continue  # yellow is not on one side: no reliable direction
        b.update(kind="redirect", color="red", head=(hx, hy),
                 heading=float(np.degrees(np.arctan2(hy - cy, hx - cx)) % 360))
        chips.append(b)

    # Green chip: a round green disc (printed text is closed by CHIP_CLOSE_PX).
    green = _clean(green_raw, CHIP_CLOSE_PX)
    for b in circle_candidates(green, green_raw, rejected, "groen: "):
        b.update(kind="split", color="green", head=None, heading=None)
        chips.append(b)
    return chips, {"red": red_raw, "yellow": yellow_raw, "green": green_raw}


class ChipTracker:
    """Gives chips a stable id and smooths position and heading.

    A new chip is only reported after TRACK_CONFIRM_FRAMES detections; a chip that
    is briefly not detected is kept for TRACK_HOLD_S seconds ("held"). When a chip
    is lifted and put down elsewhere, the held track is reused, so the game sees
    one chip moving instead of an old and a new one."""

    def __init__(self):
        self.tracks = []
        self.next_id = 1

    @staticmethod
    def _unit(deg):
        return np.cos(np.radians(deg)), np.sin(np.radians(deg))

    def update(self, detections, now):
        for kind, cfg in CHIP_KINDS.items():
            dets = sorted([d for d in detections if d["kind"] == kind], key=lambda d: d["area"], reverse=True)
            dets = dets[: cfg.get("max_objects", 1)]
            tracks = [t for t in self.tracks if t["kind"] == kind]
            used = set()
            pairs = sorted(((np.hypot(d["center"][0] - t["x"], d["center"][1] - t["y"]), i, j)
                            for i, d in enumerate(dets) for j, t in enumerate(tracks)), key=lambda p: p[0])
            matched = {}
            for dist, i, j in pairs:
                if i in matched or j in used:
                    continue
                if dist <= max(3 * tracks[j]["r"], 60):
                    matched[i] = j
                    used.add(j)
            # Unmatched detection: reuse a track of the same kind that is currently
            # missing (the chip was picked up and put down elsewhere).
            for i, d in enumerate(dets):
                if i in matched:
                    continue
                spare = [j for j, t in enumerate(tracks) if j not in used and t["last"] < now]
                if spare:
                    j = max(spare, key=lambda j: tracks[j]["hits"])
                    matched[i] = j
                    used.add(j)
                    tracks[j]["jump"] = True
            for i, d in enumerate(dets):
                if i in matched:
                    self._apply(tracks[matched[i]], d, now)
                else:
                    t = {"id": self.next_id, "kind": kind, "color": d["color"], "hits": 0, "last": now,
                         "x": d["center"][0], "y": d["center"][1], "r": d["r"], "hdx": None, "hdy": None, "jump": True}
                    self.next_id += 1
                    self._apply(t, d, now)
                    self.tracks.append(t)
        self.tracks = [t for t in self.tracks
                       if now - t["last"] <= (TRACK_HOLD_S if t["hits"] >= TRACK_CONFIRM_FRAMES else 0.25)]
        return [self._out(t, now) for t in self.tracks if t["hits"] >= TRACK_CONFIRM_FRAMES]

    def _apply(self, t, d, now):
        cx, cy = d["center"]
        dt = max(0.0, now - t["last"]) or 1 / 30
        jump = t.pop("jump", False) or np.hypot(cx - t["x"], cy - t["y"]) > t["r"]
        k = 1.0 if jump else 1 - np.exp(-dt / TRACK_POS_SMOOTH_S)
        t["x"] += (cx - t["x"]) * k
        t["y"] += (cy - t["y"]) * k
        t["r"] += (d["r"] - t["r"]) * (1.0 if jump else 0.3)
        if d.get("heading") is not None:
            ux, uy = self._unit(d["heading"])
            if t["hdx"] is None or jump:
                t["hdx"], t["hdy"] = ux, uy
            else:
                a = 1 - np.exp(-dt / TRACK_HEADING_SMOOTH_S)
                vx, vy = t["hdx"] * (1 - a) + ux * a, t["hdy"] * (1 - a) + uy * a
                n = np.hypot(vx, vy) or 1.0
                t["hdx"], t["hdy"] = vx / n, vy / n
        t["hits"] += 1
        t["last"] = now
        t["box"] = d["box"]

    def _out(self, t, now):
        heading = None if t["hdx"] is None else float(np.degrees(np.arctan2(t["hdy"], t["hdx"])) % 360)
        return {"kind": t["kind"], "color": t["color"], "id": t["id"], "center": (t["x"], t["y"]), "r": t["r"],
                "heading": heading, "held": t["last"] < now, "box": t.get("box", (0, 0, 0, 0)), "area": np.pi * t["r"] ** 2}


# ============================================================
# Coordinates: camera pixels -> normalized play area (0..1)
# ============================================================
class Mapper:
    def __init__(self, corners):
        self.corners = corners
        self.matrix = None
        if corners is not None:
            src = np.array(corners, np.float32)
            dst = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
            self.matrix = cv2.getPerspectiveTransform(src, dst)

    def raw(self, cx, cy, frame_w, frame_h):
        """Normalized coordinates without bounds check (helper points such as the chip rim)."""
        if self.matrix is None:
            return cx / frame_w, cy / frame_h
        pt = np.array([[[cx, cy]]], np.float32)
        nx, ny = cv2.perspectiveTransform(pt, self.matrix)[0][0]
        return float(nx), float(ny)

    def normalize(self, cx, cy, frame_w, frame_h):
        """Returns (nx, ny), or None if the point lies outside the play area."""
        nx, ny = self.raw(cx, cy, frame_w, frame_h)
        margin = 0.02
        if not (-margin <= nx <= 1 + margin and -margin <= ny <= 1 + margin):
            return None
        return min(1.0, max(0.0, float(nx))), min(1.0, max(0.0, float(ny)))


def build_payload(objects, mapper, frame_w, frame_h, chips=()):
    items = []
    per_color = {}
    for obj in objects:
        norm = mapper.normalize(*obj["center"], frame_w, frame_h)
        if norm is None:
            continue
        n = per_color.get(obj["color"], 0)
        per_color[obj["color"]] = n + 1
        items.append({
            # Phase 1: id = color + index (largest first). Stable identity for
            # several blocks of the same color requires tracking.
            "id": f'{obj["color"]}-{n}',
            "color": obj["color"],
            "nx": round(norm[0], 4),
            "ny": round(norm[1], 4),
            "angle": round(obj["angle"], 1),
            "area": int(obj["area"]),
            "kind": "probe",
        })
    # Chips: kind, position, rim point (gives the size on the canvas) and, for the
    # red chip, the yellow segment as a heading point. The browser maps these
    # points with its own calibration, so perspective and mirroring are included.
    for c in chips:
        norm = mapper.normalize(*c["center"], frame_w, frame_h)
        if norm is None:
            continue
        cx, cy = c["center"]
        rim = mapper.raw(cx + c["r"], cy, frame_w, frame_h)
        item = {
            "id": f'{c["color"]}-{c["id"]}',
            "kind": c["kind"],
            "color": c["color"],
            "nx": round(norm[0], 4),
            "ny": round(norm[1], 4),
            "rim": [round(rim[0], 4), round(rim[1], 4)],
            "area": int(c["area"]),
            "held": bool(c["held"]),
        }
        if c["heading"] is not None:
            a = np.radians(c["heading"])
            head = mapper.raw(cx + np.cos(a) * c["r"], cy + np.sin(a) * c["r"], frame_w, frame_h)
            item["head"] = [round(head[0], 4), round(head[1], 4)]
            item["angle"] = round(c["heading"], 1)
        items.append(item)
    primary = items[0] if items else None
    return {
        "type": "camera",
        "v": 1,
        "ts": int(time.time() * 1000),
        "detected": primary is not None,
        "nx": primary["nx"] if primary else None,
        "ny": primary["ny"] if primary else None,
        "objects": items,
    }


# ============================================================
# WebSocket
# ============================================================
clients = set()
last_message = None


async def websocket_handler(websocket, *args):
    """*args: older websockets versions also pass 'path'."""
    clients.add(websocket)
    print(f"Browser verbonden ({len(clients)} client(s)).")
    try:
        if last_message is not None:
            await websocket.send(last_message)  # send the current state right away
        async for _ in websocket:  # ignore incoming messages (for now)
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        clients.discard(websocket)
        print(f"Browserverbinding gesloten ({len(clients)} client(s)).")


async def _send(client, message):
    try:
        await asyncio.wait_for(client.send(message), timeout=0.5)
    except (websockets.exceptions.ConnectionClosed, asyncio.TimeoutError, OSError):
        clients.discard(client)


async def broadcast(data):
    global last_message
    last_message = json.dumps(data, separators=(",", ":"))
    if clients:
        await asyncio.gather(*(_send(c, last_message) for c in list(clients)))


# ============================================================
# Preview
# ============================================================
_last_hsv = {"img": None}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = _last_hsv["img"]
        extra = ""
        if hsv is not None and 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
            h, s, v = (int(n) for n in hsv[y, x])
            extra = f"  HSV=({h}, {s}, {v})"
        print(f"Klik in camerabeeld: ({x}, {y}){extra}")


def mask_preview(white_mask, chip_masks):
    """Color-coded mask: white = test object, red/yellow/green = chip colors."""
    out = cv2.cvtColor(white_mask, cv2.COLOR_GRAY2BGR) // 2
    if chip_masks:
        out[chip_masks["red"] > 0] = (40, 40, 230)
        out[chip_masks["green"] > 0] = (60, 200, 60)
        out[chip_masks["yellow"] > 0] = (0, 220, 255)
    return out


def draw_chips(frame, chips):
    for c in chips:
        cx, cy, r = int(c["center"][0]), int(c["center"][1]), int(c["r"])
        col = (40, 40, 230) if c["kind"] == "redirect" else (60, 200, 60)
        cv2.circle(frame, (cx, cy), r + 4, (255, 255, 255) if not c["held"] else (160, 160, 160), 2)
        cv2.circle(frame, (cx, cy), r, col, 3)
        label = "splits" if c["kind"] == "split" else "stroom"
        if c["heading"] is not None:
            a = np.radians(c["heading"])
            tip = (int(cx + np.cos(a) * (r + 40)), int(cy + np.sin(a) * (r + 40)))
            cv2.arrowedLine(frame, (cx, cy), tip, (0, 220, 255), 3, tipLength=0.25)
            label += f' {c["heading"]:.0f} deg'
        if c["held"]:
            label += " (vastgehouden)"
        cv2.putText(frame, f'#{c["id"]} {label}', (cx - r, max(20, cy - r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


def draw_preview(frame, objects, payload, mapper, fps, n_clients, rejected=()):
    for r in rejected:
        x, y, w, h = r["box"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'genegeerd: {r["reason"]}', (x + 6, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if mapper.corners is not None:
        pts = np.array(mapper.corners, np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
    for obj in objects:
        x, y, w, h = obj["box"]
        cx, cy = int(obj["center"][0]), int(obj["center"][1])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.circle(frame, (cx, cy), 7, (0, 0, 255), -1)
        cv2.putText(frame, f'{obj["color"]} {obj["angle"]:.0f} deg', (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    if payload["detected"]:
        info = f'nx={payload["nx"]:.3f} ny={payload["ny"]:.3f}'
    else:
        info = "geen object"
    cv2.putText(frame, info, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"{fps:4.1f} fps | browsers: {n_clients} | Q/Esc = stop",
                (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


_window_seen = False


def window_closed():
    """True if the user closed the camera window with its close button."""
    global _window_seen
    try:
        visible = cv2.getWindowProperty(WINDOW_CAMERA, cv2.WND_PROP_VISIBLE)
    except cv2.error:
        return _window_seen
    if visible >= 1:
        _window_seen = True
        return False
    return _window_seen  # some backends never report 'visible'


# ============================================================
# Main loop
# ============================================================
async def main(args):
    print(f"OpenCV {cv2.__version__} | websockets {websockets.__version__} | Python {sys.version.split()[0]}")

    reader = CameraReader(args.camera, FRAME_WIDTH, FRAME_HEIGHT)
    print(f"Camera {args.camera} openen...")
    if not reader.start():
        print("FOUT:", reader.error)
        reader.stop()
        return 1
    print(f"Camera actief via {reader.backend}, resolutie {reader.size[0]}x{reader.size[1]}.")
    if (FRAME_WIDTH, FRAME_HEIGHT) != reader.size:
        print("  Let op: de camera levert een andere resolutie dan gevraagd. "
              "MIN_AREA/MAX_AREA en TABLE_CORNERS gelden voor de werkelijke resolutie.")

    mapper = Mapper(TABLE_CORNERS)
    print(f"Oriëntatie: spiegelen links/rechts={'aan' if args.mirror else 'uit'}, "
          f"boven/onder={'aan' if args.flip_y else 'uit'}, "
          f"speelveld={'TABLE_CORNERS' if TABLE_CORNERS else 'volledig camerabeeld'}.")

    try:
        server = await websockets.serve(websocket_handler, HOST, args.port)
    except OSError as e:
        print(f"FOUT: kan WebSocket-server niet starten op {HOST}:{args.port} ({e}).\n"
              "  Draait camera_bridge.py al in een ander venster? Of kies --port.")
        reader.stop()
        return 1

    print(f"WebSocket-server: ws://{HOST}:{args.port}")
    print("Druk op Q of Esc in het cameravenster (of Ctrl+C hier) om te stoppen.")

    preview = not args.no_preview
    if preview:
        cv2.namedWindow(WINDOW_CAMERA, cv2.WINDOW_NORMAL)
        cv2.namedWindow(WINDOW_MASK, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_CAMERA, on_mouse)

    global CHIP_DETECTION
    if args.no_chips:
        CHIP_DETECTION = False
    print(f"Fiche-detectie: {'aan (rood = stroming wijzigen, groen = splitsen)' if CHIP_DETECTION else 'uit'}.")
    tracker = ChipTracker()
    send_interval = 1.0 / SEND_HZ
    last_debug = 0.0
    last_send = 0.0
    last_detected = None
    seq = 0
    fps, fps_t, fps_n = 0.0, time.perf_counter(), 0
    exit_code = 0

    try:
        while True:
            new_seq, frame = await asyncio.to_thread(reader.wait_frame, seq, 1.0)
            if not reader.alive:
                print("FOUT:", reader.error or "camerathread gestopt.")
                exit_code = 1
                break
            if frame is None or new_seq == seq:
                if preview and cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue  # no new frame yet
            seq = new_seq
            if args.mirror and args.flip_y:
                frame = cv2.flip(frame, -1)
            elif args.mirror:
                frame = cv2.flip(frame, 1)
            elif args.flip_y:
                frame = cv2.flip(frame, 0)

            frame_h, frame_w = frame.shape[:2]
            rejected = []
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _last_hsv["img"] = hsv
            objects, mask = detect_objects(frame, rejected)
            chips, chip_masks = [], None
            if CHIP_DETECTION:
                found, chip_masks = detect_chips(hsv, rejected)
                chips = tracker.update(found, time.perf_counter())
            payload = build_payload(objects, mapper, frame_w, frame_h, chips)

            if args.debug and time.perf_counter() - last_debug >= 0.5:
                last_debug = time.perf_counter()
                if payload["detected"]:
                    o = objects[0]
                    print(f'[debug] pixel=({o["center"][0]:.0f},{o["center"][1]:.0f}) van {frame_w}x{frame_h}'
                          f' -> nx={payload["nx"]:.3f} ny={payload["ny"]:.3f}'
                          f' | kandidaten={len(objects)} genegeerd={len(rejected)}')
                else:
                    print(f'[debug] geen object | genegeerd={len(rejected)}')
                for c in chips:
                    hd = f' richting={c["heading"]:.0f} gr' if c["heading"] is not None else ''
                    print(f'[debug] fiche #{c["id"]} {c["kind"]}: pixel=({c["center"][0]:.0f},{c["center"][1]:.0f})'
                          f' straal={c["r"]:.0f}px{hd}{" (vastgehouden)" if c["held"] else ""}')

            now = time.perf_counter()
            # Send state changes (which objects are present) immediately, otherwise rate-limited.
            present = tuple(o["id"] for o in payload["objects"])
            if present != last_detected or now - last_send >= send_interval:
                await broadcast(payload)
                last_send = now
                last_detected = present

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0

            if preview:
                draw_preview(frame, objects, payload, mapper, fps, len(clients), rejected)
                draw_chips(frame, chips)
                cv2.imshow(WINDOW_CAMERA, frame)
                cv2.imshow(WINDOW_MASK, mask_preview(mask, chip_masks))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or window_closed():
                    break
            else:
                await asyncio.sleep(0)
    finally:
        # Tell browsers cleanly that nothing is detected anymore.
        try:
            await broadcast({"type": "camera", "v": 1, "ts": int(time.time() * 1000),
                             "detected": False, "nx": None, "ny": None, "objects": []})
        except Exception:
            pass
        server.close()
        await server.wait_closed()
        reader.stop()
        if preview:
            cv2.destroyAllWindows()
        print("Camera-server gestopt.")
    return exit_code


def parse_args():
    p = argparse.ArgumentParser(description="People Manager camera bridge")
    p.add_argument("--camera", type=int, default=CAMERA_INDEX, help="camera-index (0, 1, ...)")
    p.add_argument("--port", type=int, default=PORT, help="WebSocket-poort")
    p.add_argument("--mirror", action="store_true", default=MIRROR_X, help="beeld horizontaal spiegelen")
    p.add_argument("--flip-y", action="store_true", default=FLIP_Y, help="beeld verticaal spiegelen")
    p.add_argument("--debug", action="store_true", help="pixel- en genormaliseerde coördinaten in de terminal tonen")
    p.add_argument("--no-chips", action="store_true", help="fiche-detectie uitschakelen (alleen het witte testobject)")
    p.add_argument("--no-preview", action="store_true", default=not SHOW_PREVIEW,
                   help="geen OpenCV-vensters tonen")
    return p.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(parse_args())))
    except KeyboardInterrupt:
        print("Camera-server gestopt.")
