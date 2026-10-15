import argparse
import asyncio
import json
import os
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

# Exposure: how much light the webcam collects. With automatic exposure the webcam
# brightens the image until the bright monitor and the chips clip to (almost) white,
# which destroys exactly the colors the detection needs. Manual exposure keeps them.
# Adjust live in the camera window with [ and ] (or - and +); the last values are
# saved in camera_settings.json and used on the next start.
EXPOSURE_AUTO = False     # True = let the webcam decide (old behavior)
EXPOSURE = -6             # Windows (DirectShow/MSMF): 2^EXPOSURE seconds; -6 = 1/64 s. Lower = darker
GAIN = 0                  # amplification; 0 = none (least noise). None = leave the webcam setting alone
BRIGHTNESS = 1.4          # fine adjustment in software (x1.0 .. x3.0, steps of 0.1 with , and .):
                          # between two exposure steps; safe because nothing is clipped at the sensor
SETTINGS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_settings.json")
CLIPPED_WARN = 0.05       # warn in the preview when more than 5% of the pixels are overexposed

HOST = "127.0.0.1"
PORT = 8765
SEND_HZ = 30              # maximum number of messages per second to the browser

# Original detection values for the white test object.
BRIGHTNESS_MIN = 200
SATURATION_MAX = 145
MIN_AREA = 3000
MAX_AREA = 150000
SOLIDITY_MIN = 0.70
# White blocks (a strip of paper): a wall in the game. People bump into it and walk
# around it, then continue in their own direction. Shape checks on the rotated rectangle,
# so a block may lie at any angle.
BLOCK_ASPECT_MAX = 8.0     # long side / short side
BLOCK_RECT_MIN = 0.75      # area / area of the rotated bounding rectangle (it must be rectangular)

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
        "max_objects": 4,
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
# Red chip: first the whole warm chip (red, orange and yellow together) is found with
# low thresholds; then the yellow segment is the part that is clearly more yellow
# than the rest of that chip. This is relative, so it keeps working when the webcam
# makes the image darker, warmer (yellow turns orange) or less saturated.
WARM_RANGES = [((0, 0, 40), (40, 255, 255)), ((165, 0, 40), (180, 255, 255))]  # saturation: see below
# Minimum saturation of the warm chip, chosen per frame: the split (Otsu) between the
# vivid chip and duller warm surfaces such as wood, kept within these limits.
WARM_S_LIMITS = (40, 140)
RED_HUE_MAX = 16      # the red body (often orange-red under a webcam): hue at most this (OpenCV hue 0..180; 165..180 counts as just below 0)
HUE_GAP_MIN = 8       # the yellow segment must be at least this much more yellow than the body
GREEN_RANGES = [((55, 0, 40), (96, 255, 255))]  # wide: webcams shift green toward blue-green; saturation: see below
GREEN_S_LIMITS = (50, 140)  # per-frame minimum saturation of the green chip, like WARM_S_LIMITS

CHIP_MIN_RADIUS = 12       # px in the camera frame; print with --debug to tune
CHIP_MAX_RADIUS = 250      # px (a chip close to the camera is large)
CHIP_FILL_MIN = 0.60       # share of colored pixels inside the chip circle (rejects rings and outlines)
CHIP_SURROUND_MAX = 0.35   # max. share of the same color in the ring just outside the chip (rejects bars and areas)
RED_MIN_FRACTION = 0.35    # red share of the colored part of a red chip
YELLOW_FRACTION = (0.04, 0.55)  # yellow share of the colored part: present, but not the whole chip
GLARE_MAX = 0.75           # max. share of the chip hidden by glare (reflection on tape or plastic)
RED_VIVID_MIN = 0.25       # a red chip always shows at least this much truly vivid red, even with much glare
                           # (pale, light areas such as the cream tray cards do not)
HOLE_FILL_MAX = 0.50       # holes enclosed by a chip (glare, tape, printed text) are part of the chip,
                           # up to this share of its area; a thin ring (cursor ring) has a bigger hole
HULL_FILL_MIN = 0.35       # a blob bitten by glare at its edge is completed to its convex outline when it
                           # already fills this share of it
HULL_MIN_RADIUS = 20       # px; only chips this big get the glare completion (glare matters on big chips)
GLARE_BRIGHTNESS = 0.85    # glare is at least this bright relative to the chip color around it
CHIP_INSIDE_S_MIN = 60     # saturation limit used inside a found chip to split red and yellow
GLARE_S_MIN = 30           # glare on a chip keeps at least this much of its tint
GLARE_STREAK_MAX = 0.15    # a pure white glare streak is accepted up to this share of the chip
HULL_NEARLY_CONVEX = 0.90  # a blob filling this much of its convex outline is completed without further checks
CHIP_SIZE_RATIO_MIN = 0.6  # all chips are the same size: a circle much smaller than the biggest chip
                           # in view is a piece of a chip (or something else), not a chip
CHIP_SIZE_MEMORY_S = 30    # the chip size seen is remembered this long (also when no chip is in view)
CHIP_CLOSE_PX = 9          # closes gaps caused by reflections, tape and printed text
# Speed: once chips are known, detection runs on a reduced image that still gives the
# smallest known chip DETECT_RADIUS_PX pixels of radius. Without chips: full resolution.
DETECT_RADIUS_PX = 24
DETECT_SCALE_MIN = 0.35

# How many chips of each kind may be active at the same time (largest first).
CHIP_KINDS = {
    "redirect": {"color": "red", "max_objects": 4},
    "split": {"color": "green", "max_objects": 4},
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
        self.api = ""             # OpenCV backend name (DSHOW, MSMF, V4L2, ...)
        self.controls = {}        # exposure/gain as reported back by the webcam
        self._pending = {}
        self._ctl_lock = threading.Lock()
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
                    try:
                        self.api = cap.getBackendName()
                    except Exception:
                        self.api = name
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
        self._apply_controls(cap)  # initial exposure settings, before the first published frame
        self._publish(frame)
        self._ready.set()

        failures = 0
        try:
            while not self._stop.is_set():
                self._apply_controls(cap)
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

    # ---- Exposure / gain (applied in the camera thread: drivers expect that) ----
    def request(self, **changes):
        """Queue control changes: auto (bool), exposure (number), gain (number), dialog (True)."""
        with self._ctl_lock:
            self._pending.update(changes)

    def _exposure_value(self, e):
        # The configured value is the Windows log2 scale. V4L2 (Linux) uses units of 100 µs.
        if self.api == "V4L2" and e <= 0:
            return max(1, round(2.0 ** e * 10000))
        return e

    def _apply_controls(self, cap):
        with self._ctl_lock:
            p, self._pending = self._pending, {}
        if not p:
            return
        try:
            self._set_controls(cap, p)
        except Exception as e:  # a driver that misbehaves must never stop the camera
            print(f"Let op: belichting instellen mislukt ({e}).")

    def _set_controls(self, cap, p):
        res = dict(self.controls.get("accepted", {}))
        if "auto" in p:
            want_auto = bool(p["auto"])
            # 0.25 = manual and 0.75 = auto is how OpenCV passes the mode to most drivers;
            # 1/3 are the V4L2 values. Not every webcam supports switching.
            vals = (0.75, 3) if want_auto else (0.25, 1)
            res["auto"] = any(cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, v) for v in vals)
            self.controls["auto"] = want_auto
        if "exposure" in p and not self.controls.get("auto", False):
            res["exposure"] = cap.set(cv2.CAP_PROP_EXPOSURE, self._exposure_value(p["exposure"]))
            self.controls["exposure_set"] = p["exposure"]
        if p.get("gain") is not None:
            res["gain"] = cap.set(cv2.CAP_PROP_GAIN, p["gain"])
        if p.get("dialog"):
            res["dialog"] = cap.set(cv2.CAP_PROP_SETTINGS, 1)  # driver settings window (DirectShow on Windows)
        self.controls["accepted"] = res
        self.controls["exposure"] = cap.get(cv2.CAP_PROP_EXPOSURE)
        self.controls["gain"] = cap.get(cv2.CAP_PROP_GAIN)

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
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0:
            continue
        solidity = area / hull_area
        rect = cv2.minAreaRect(contour)
        pts = cv2.boxPoints(rect)
        e0, e1 = pts[1] - pts[0], pts[2] - pts[1]
        l0, l1 = float(np.hypot(*e0)), float(np.hypot(*e1))
        long_side, short_side = max(l0, l1), max(1.0, min(l0, l1))
        long_vec = e0 if l0 >= l1 else e1
        if long_side / short_side > BLOCK_ASPECT_MAX or solidity < SOLIDITY_MIN:
            continue
        if area / (long_side * short_side) < BLOCK_RECT_MIN:
            if rejected is not None:
                rejected.append({"box": (int(x), int(y), int(w), int(h)), "reason": "wit: geen rechthoek"})
            continue
        if REJECT_BORDER_TOUCHING and (
            x <= BORDER_MARGIN or y <= BORDER_MARGIN
            or x + w >= mask_w - BORDER_MARGIN or y + h >= mask_h - BORDER_MARGIN
        ):
            if rejected is not None:
                rejected.append({"box": (int(x), int(y), int(w), int(h)), "reason": "raakt rand"})
            continue

        # Centroid: more accurate than the bounding-box center
        # once a block lies at an angle.
        mom = cv2.moments(contour)
        if mom["m00"]:
            cx, cy = mom["m10"] / mom["m00"], mom["m01"] / mom["m00"]
        else:
            cx, cy = x + w / 2.0, y + h / 2.0

        # Orientation of the long side (0..180 degrees; a rectangle has no front or back).
        theta = float(np.degrees(np.arctan2(long_vec[1], long_vec[0])) % 180.0)
        rcx, rcy = rect[0]

        candidates.append({
            "box": (int(x), int(y), int(w), int(h)),
            "area": area,
            "center": (cx, cy),
            "angle": theta,
            "rect": (float(rcx), float(rcy), long_side, short_side, theta),
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
_scale = {"k": 1.0}  # processing scale of the current detect_chips call (pixel sizes scale with it)


def _px(v):
    return v * _scale["k"]


def _kernel_px(v):
    return max(3, int(round(v * _scale["k"])) | 1)  # odd kernel size, at least 3


def ranges_mask(hsv, ranges):
    mask = None
    for lower, upper in ranges:
        m = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    return mask


def _glare_like(pixels_v, pixels_tint, blob_v):
    """Glare on a chip is light and keeps a tint of the chip's own color (pink-white on
    the red chip). Dark wood grain or the bluish-grey game floor do not qualify."""
    return pixels_tint >= 0.5 and pixels_v >= GLARE_BRIGHTNESS * blob_v


def _fill_holes(mask, tint, vch):
    """Fill holes that are completely enclosed by a blob, look like glare (see _glare_like)
    and are not larger than HOLE_FILL_MAX of the blob. Returns (filled mask, filled holes)."""
    inv = cv2.bitwise_not(mask)
    h, w = mask.shape
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 0)  # remove the outside; what remains are holes
    holes_all = ff
    if not cv2.countNonZero(holes_all):
        return mask, np.zeros_like(mask)
    n_h, lab_h = cv2.connectedComponents(holes_all)
    whole = cv2.bitwise_or(mask, holes_all)
    n_b, lab_b = cv2.connectedComponents(whole)
    lh, lb = lab_h.ravel(), lab_b.ravel()
    v = vch.ravel()
    ih = np.flatnonzero(lh)                  # hole pixels only (fast: few pixels)
    im = np.flatnonzero(mask.ravel())        # chip-colored pixels only
    hole_px = np.bincount(lh[ih], minlength=n_h)
    hole_v = np.bincount(lh[ih], weights=v[ih], minlength=n_h) / np.maximum(hole_px, 1)
    hole_t = np.bincount(lh[ih], weights=(tint.ravel()[ih] > 0), minlength=n_h) / np.maximum(hole_px, 1) if tint is not None else None
    blob_px = np.bincount(lb[im], minlength=n_b)
    blob_v = np.bincount(lb[im], weights=v[im], minlength=n_b) / np.maximum(blob_px, 1)
    whole_px = blob_px + np.bincount(lb[ih], minlength=n_b)
    parent = np.zeros(n_h, np.int64)
    parent[lh[ih]] = lb[ih]  # every hole lies inside exactly one blob of `whole`
    keep = hole_px <= HOLE_FILL_MAX * whole_px[parent]
    if tint is not None:  # only glare-like holes (red chip: dark wood grain must not be filled)
        keep &= (hole_t >= 0.5) & (hole_v >= GLARE_BRIGHTNESS * blob_v[parent])
    keep[0] = False
    holes = (keep[lab_h] & (lab_h > 0)).astype(np.uint8) * 255
    return cv2.bitwise_or(mask, holes), holes


def _hull_filled(mask, tint, vch):
    """Blobs completed to their convex outline when they already fill HULL_FILL_MIN of it
    and the missing part looks like glare: a chip with a glare streak running to its
    edge becomes whole again; a thin ring, an L-shape or a cone next to a stripe do not."""
    out = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area < np.pi * _px(HULL_MIN_RADIUS) ** 2:
            continue
        hull = cv2.convexHull(c)
        fill_ratio = area / max(1.0, cv2.contourArea(hull))
        if fill_ratio < HULL_FILL_MIN:
            continue
        if fill_ratio >= HULL_NEARLY_CONVEX:
            cv2.drawContours(out, [hull], -1, 255, -1)  # nearly its own outline already: just close notches and streaks
            continue
        x, y, w, h = cv2.boundingRect(hull)
        hm = np.zeros((h, w), np.uint8)
        cv2.drawContours(hm, [hull - (x, y)], -1, 255, -1)
        blob = mask[y:y + h, x:x + w] > 0
        gap = (hm > 0) & ~blob
        if gap.any():
            vv = vch[y:y + h, x:x + w]
            gap_v, blob_v = float(vv[gap].mean()), float(vv[blob].mean())
            small_white = gap.sum() <= GLARE_STREAK_MAX * (hm > 0).sum() and gap_v >= GLARE_BRIGHTNESS * blob_v
            if not small_white and not _glare_like(gap_v, float((tint[y:y + h, x:x + w][gap] > 0).mean()), blob_v):
                continue
        cv2.drawContours(out, [hull], -1, 255, -1)
    return out


def _candidates(mask, raw, holes, rejected, label, tint, vch):
    """Chip circles from the mask itself and, for chips bitten by glare, from the
    hull-filled mask; duplicates of the same chip are merged (the direct one wins)."""
    found = circle_candidates(mask, cv2.bitwise_or(raw, holes), rejected, label)
    hulled = _hull_filled(mask, tint, vch)
    extra = circle_candidates(hulled, cv2.bitwise_or(cv2.bitwise_or(raw, holes), cv2.bitwise_and(hulled, cv2.bitwise_not(mask))), None, label)
    for e in extra:
        if all(np.hypot(e["center"][0] - f["center"][0], e["center"][1] - f["center"][1]) > 0.25 * max(e["r"], f["r"]) or
               abs(e["r"] - f["r"]) > 0.25 * max(e["r"], f["r"]) for f in found):
            e["hull"] = True
            found.append(e)  # overlapping candidates are resolved after validation (_largest_first)
    return found


def _largest_first(chips):
    """Of overlapping valid chips keep the largest: a small circle inside a big chip
    (part of its rim next to glare) is not a chip of its own. Chips much smaller than the
    biggest one are dropped as well (all physical chips have the same size)."""
    if chips:
        biggest = max(c["r"] for c in chips)
        chips = [c for c in chips if c["r"] >= CHIP_SIZE_RATIO_MIN * biggest]
    kept = []
    for c in sorted(chips, key=lambda c: c["r"], reverse=True):
        if all(np.hypot(c["center"][0] - k["center"][0], c["center"][1] - k["center"][1]) > 0.9 * k["r"] for k in kept):
            kept.append(c)
    return kept


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
    for _ in range(12):
        _, r, _, (px, py) = cv2.minMaxLoc(dt)
        if r < _px(CHIP_MIN_RADIUS) * 0.8:
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
        # Only solid areas count around the chip, not loose specks (noise, small game graphics).
        solid = cv2.morphologyEx(mask[wy0:wy1, wx0:wx1], cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(r / 4) | 1),) * 2))
        around = np.count_nonzero(solid[ring]) / max(1, np.count_nonzero(ring))
        reason = None
        if r > _px(CHIP_MAX_RADIUS):
            reason = "te groot"
        elif fill < CHIP_FILL_MIN:
            reason = "niet gevuld"
        elif around > CHIP_SURROUND_MAX:
            reason = "niet rond (deel van groter vlak)"
        elif REJECT_BORDER_TOUCHING and (cx - r <= _px(BORDER_MARGIN) or cy - r <= _px(BORDER_MARGIN) or
                                         cx + r >= w_img - _px(BORDER_MARGIN) or cy + r >= h_img - _px(BORDER_MARGIN)):
            reason = "raakt rand"
        if reason:
            if rejected is not None:
                rejected.append({"box": box, "reason": f"{label}{reason}"})
            continue
        found.append({"center": (float(cx), float(cy)), "r": float(r), "area": float(np.pi * r * r), "box": box})
    return found


def _refine_local(b, hsv, warm_hue, s_min):
    """Re-determine center and radius of a red-chip candidate with a saturation split
    made only around the candidate: wood or other warm surfaces right next to the chip
    are duller than the chip itself and should not shift its center."""
    h_img, w_img = hsv.shape[:2]
    cx, cy, r = b["center"][0], b["center"][1], b["r"]
    R = int(r * 1.8) + 3
    x0, x1, y0, y1 = max(0, int(cx) - R), min(w_img, int(cx) + R + 1), max(0, int(cy) - R), min(h_img, int(cy) + R + 1)
    sat = hsv[y0:y1, x0:x1, 1]
    hue_ok = warm_hue[y0:y1, x0:x1] > 0
    vals = sat[hue_ok & (sat >= 20)]
    if vals.size < 50:
        return
    t, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    loc = (hue_ok & (sat >= max(t, s_min))).astype(np.uint8) * 255
    loc, _ = _fill_holes(_clean(loc, _kernel_px(CHIP_CLOSE_PX)), cv2.bitwise_and(warm_hue[y0:y1, x0:x1], cv2.inRange(sat, 15, 255)), hsv[y0:y1, x0:x1, 2])
    dt = cv2.distanceTransform(loc, cv2.DIST_L2, 5)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    near = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r  # the same chip, not a neighbor
    dt_near = np.where(near, dt, 0)
    r2 = float(dt_near.max())
    if r2 < max(_px(CHIP_MIN_RADIUS) * 0.8, r * 0.6):
        return  # refinement would throw the chip away: keep the original estimate
    ys, xs = np.nonzero(dt_near >= r2 * 0.9)
    b["center"] = (float(x0 + xs.mean()), float(y0 + ys.mean()))
    b["r"] = r2
    b["area"] = float(np.pi * r2 * r2)
    b["box"] = (int(b["center"][0] - r2), int(b["center"][1] - r2), int(2 * r2), int(2 * r2))


_warm_s = {"value": None}
_green_s = {"value": None}


def _adaptive_saturation_min(hsv, hue_mask, limits, state):
    """Minimum saturation that separates a vivid chip from duller surfaces of a similar
    hue in this frame (wood, skin, the game's planks; blue that turns cyan when the
    image is too bright). Otsu split, only used when the frame really contains two
    groups; smoothed over frames so the threshold does not jump."""
    sat = hsv[::2, ::2, 1]  # every 4th pixel is plenty for the statistics
    s_vals = sat[(hue_mask[::2, ::2] > 0) & (sat >= 20)]
    target = float(limits[0])
    if s_vals.size >= 125:
        t, _ = cv2.threshold(s_vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lo, hi = s_vals[s_vals < t], s_vals[s_vals >= t]
        # Glare on the chip may fall below this split; the glare completion puts it back.
        if lo.size and hi.size and float(hi.mean()) - float(lo.mean()) >= 40:
            target = float(np.clip(t, *limits))
    prev = state["value"]
    state["value"] = target if prev is None else prev + (target - prev) * 0.2
    return int(round(state["value"]))


def _warm_saturation_min(hsv, warm_hue):
    return _adaptive_saturation_min(hsv, warm_hue, WARM_S_LIMITS, _warm_s)


def detect_chips(hsv, rejected=None, bgr=None, scale=1.0):
    """Chip detection, optionally on a reduced image (`scale` < 1) for speed. Positions,
    radii and boxes in the result are always in full-resolution camera pixels."""
    if scale >= 0.999:
        _scale["k"] = 1.0
        return _detect_chips(hsv, rejected, bgr)
    if bgr is None:
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    small = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)  # resize BGR, not HSV (hue wraps)
    _scale["k"] = scale
    rej = [] if rejected is not None else None
    try:
        chips, masks = _detect_chips(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), rej, small)
    finally:
        _scale["k"] = 1.0
    f = 1.0 / scale
    for c in chips:
        c["center"] = (c["center"][0] * f, c["center"][1] * f)
        c["r"] *= f
        c["area"] *= f * f
        c["box"] = tuple(int(v * f) for v in c["box"])
        if c.get("head") is not None:
            c["head"] = (c["head"][0] * f, c["head"][1] * f)
    if rejected is not None:
        for r in rej:
            r["box"] = tuple(int(v * f) for v in r["box"])
            rejected.append(r)
    full = (hsv.shape[1], hsv.shape[0])
    masks = {k: cv2.resize(m, full, interpolation=cv2.INTER_NEAREST) for k, m in masks.items()}
    return chips, masks


def _detect_chips(hsv, rejected=None, bgr=None):
    """Returns (chips, masks). Each chip: kind, color, center, r (px), heading
    (degrees in the camera image, redirect only), head (yellow centroid, px).
    `bgr` (optional): the frame, used to skip washed-out pixels."""
    if bgr is None:
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Washed-out pixels (two or more color channels at the maximum) have lost their
    # real color: overexposed blue turns cyan and would look like the green chip.
    # They are not used for the green chip.
    b, g, r = cv2.split(bgr)
    middle = cv2.max(cv2.min(b, g), cv2.min(cv2.max(b, g), r))  # middle channel value; >= 250 means 2+ channels clipped
    valid = cv2.inRange(middle, 0, 249)
    warm_hue = ranges_mask(hsv, WARM_RANGES)  # bright yellow (red+green clipped) stays yellow: no washed-out rule here
    s_min = _warm_saturation_min(hsv, warm_hue)
    warm_raw = cv2.bitwise_and(warm_hue, cv2.inRange(hsv[..., 1], s_min, 255))
    red_vivid = cv2.bitwise_and(warm_raw, cv2.bitwise_or(cv2.inRange(hsv[..., 0], 0, RED_HUE_MAX),
                                                         cv2.inRange(hsv[..., 0], 150, 180)))  # truly vivid red
    # Glare on the red chip is pink: warm, less saturated, and among the brightest pixels.
    # It belongs to the chip body (but is not used as chip color for the red/yellow split).
    v_warm = hsv[..., 2][warm_raw > 0]
    if v_warm.size >= 50:
        bright = int(GLARE_BRIGHTNESS * np.percentile(v_warm, 90))
        glare = cv2.bitwise_and(warm_hue, cv2.bitwise_and(cv2.inRange(hsv[..., 1], GLARE_S_MIN, 255),
                                                          cv2.inRange(hsv[..., 2], bright, 255)))
        warm_raw = cv2.bitwise_or(warm_raw, glare)
    green_hue = cv2.bitwise_and(ranges_mask(hsv, GREEN_RANGES), valid)
    green_raw = cv2.bitwise_and(green_hue, cv2.inRange(hsv[..., 1], _adaptive_saturation_min(hsv, green_hue, GREEN_S_LIMITS, _green_s), 255))
    hue_all = hsv[..., 0].astype(np.int16)
    np.subtract(hue_all, 180, out=hue_all, where=hue_all >= 150)  # 165..180 (red) becomes -15..0, so red is one range around 0
    chips = []

    # Red chip: the warm pixels together form the round chip.
    vch = hsv[..., 2]
    tinted = cv2.inRange(hsv[..., 1], 15, 255)  # glare keeps a little of the chip's color
    warm_tint = cv2.bitwise_and(warm_hue, tinted)
    warm, warm_holes = _fill_holes(_clean(warm_raw, _kernel_px(CHIP_CLOSE_PX)), warm_tint, vch)
    h_img, w_img = hsv.shape[:2]
    for b in _candidates(warm, warm_raw, warm_holes, rejected, "rood: ", warm_tint, vch):
        if not b.get("hull"):
            _refine_local(b, hsv, warm_hue, s_min)
        cx, cy = b["center"]
        R = int(b["r"] * 1.25) + 2  # the yellow segment may stick out of the red disc
        x0, x1, y0, y1 = max(0, int(cx) - R), min(w_img, int(cx) + R + 1), max(0, int(cy) - R), min(h_img, int(cy) + R + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        disc, wide = d2 <= b["r"] ** 2, d2 <= (b["r"] * 1.25) ** 2
        disc_px = max(1, np.count_nonzero(disc))
        hue = hue_all[y0:y1, x0:x1]
        # Inside a found chip a low saturation limit is used: a pale yellow segment (glare) must
        # count, and there is no wood inside the chip to keep out.
        sel = (warm_hue[y0:y1, x0:x1] > 0) & (hsv[y0:y1, x0:x1, 1] >= CHIP_INSIDE_S_MIN) & wide
        vals = hue[sel]
        why = None
        if vals.size < 20:
            why = "te weinig kleur"
        else:
            # Split the chip's hues in two groups (Otsu): body (red) and segment (yellow).
            scaled = np.clip((vals + 20) * 4, 0, 255).astype(np.uint8).reshape(-1, 1)
            t, _ = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            thr = t / 4.0 - 20
            body, seg = sel & (hue <= thr), sel & (hue > thr)
            body_hue = float(hue[body].mean()) if body.any() else 99.0
            seg_hue = float(hue[seg].mean()) if seg.any() else -99.0
            colored = max(1, np.count_nonzero(sel))
            red_frac = np.count_nonzero(body) / colored   # shares of the colored part: glare does not count
            yel_frac = np.count_nonzero(seg) / colored
            glare = 1 - np.count_nonzero(sel & disc) / disc_px
            vivid = np.count_nonzero(red_vivid[y0:y1, x0:x1][disc]) / disc_px
            if glare > GLARE_MAX:
                why = f"te veel glans ({glare:.0%})"
            elif vivid < RED_VIVID_MIN:
                why = f"te weinig fel rood ({vivid:.0%})"
            elif body_hue > RED_HUE_MAX:
                why = f"niet rood (tint {body_hue:.0f})"
            elif seg_hue - body_hue < HUE_GAP_MIN:
                why = f"geen geel segment (tintverschil {seg_hue - body_hue:.0f})"
            elif red_frac < RED_MIN_FRACTION:
                why = f"te weinig rood ({red_frac:.0%})"
            elif not YELLOW_FRACTION[0] <= yel_frac <= YELLOW_FRACTION[1]:
                why = f"geel segment te {'klein' if yel_frac < YELLOW_FRACTION[0] else 'groot'} ({yel_frac:.0%})"
        if why:
            if rejected is not None:
                rejected.append({"box": b["box"], "reason": f"rood: {why}"})
            continue
        ys, xs = np.nonzero(seg)
        hx, hy = x0 + xs.mean(), y0 + ys.mean()
        if np.hypot(hx - cx, hy - cy) < b["r"] * 0.15:
            if rejected is not None:
                rejected.append({"box": b["box"], "reason": "rood: richting onduidelijk"})
            continue  # yellow is not on one side: no reliable direction
        b.update(kind="redirect", color="red", head=(hx, hy),
                 heading=float(np.degrees(np.arctan2(hy - cy, hx - cx)) % 360))
        chips.append(b)

    # Green chip: a round green disc (printed text is closed by CHIP_CLOSE_PX).
    green_tint = cv2.bitwise_and(green_hue, tinted)
    green, green_holes = _fill_holes(_clean(green_raw, _kernel_px(CHIP_CLOSE_PX)), None, vch)  # printed text: always part of the chip
    for b in _candidates(green, green_raw, green_holes, rejected, "groen: ", green_tint, vch):
        b.update(kind="split", color="green", head=None, heading=None)
        chips.append(b)
    hue8 = hsv[..., 0]  # for the mask window only
    red_view = cv2.bitwise_and(warm_raw, cv2.bitwise_or(cv2.inRange(hue8, 0, RED_HUE_MAX), cv2.inRange(hue8, 150, 180)))
    yellow_view = cv2.bitwise_and(warm_raw, cv2.inRange(hue8, RED_HUE_MAX + 4, 40))
    return _largest_first(chips), {"red": red_view, "yellow": yellow_view, "green": green_raw}


class ChipTracker:
    """Gives chips a stable id and smooths position and heading.

    A new chip is only reported after TRACK_CONFIRM_FRAMES detections; a chip that
    is briefly not detected is kept for TRACK_HOLD_S seconds ("held"). When a chip
    is lifted and put down elsewhere, the held track is reused, so the game sees
    one chip moving instead of an old and a new one."""

    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.size = None      # typical chip radius seen recently (all chips are the same size)
        self.size_t = -1e9

    @staticmethod
    def _unit(deg):
        return np.cos(np.radians(deg)), np.sin(np.radians(deg))

    def update(self, detections, now):
        if self.size is not None and now - self.size_t < CHIP_SIZE_MEMORY_S:
            # much smaller than the chips seen before: noise or a game element, not a chip
            detections = [d for d in detections if d["r"] >= CHIP_SIZE_RATIO_MIN * self.size]
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
        live = [t["r"] for t in self.tracks if t["hits"] >= TRACK_CONFIRM_FRAMES and t["last"] == now]
        if live:
            r = float(np.median(live))
            self.size = r if self.size is None else self.size + (r - self.size) * 0.1
            self.size_t = now
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


class BlockTracker:
    """Stable white blocks: same confirm/hold rules as the chips, smoothed rectangle
    (center, sides, orientation with 180-degree symmetry)."""

    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, detections, now):
        dets = [d for d in detections if d.get("rect")]
        used, matched = set(), {}
        pairs = sorted(((np.hypot(d["rect"][0] - t["cx"], d["rect"][1] - t["cy"]), i, j)
                        for i, d in enumerate(dets) for j, t in enumerate(self.tracks)), key=lambda p: p[0])
        for dist, i, j in pairs:
            if i in matched or j in used:
                continue
            if dist <= max(0.5 * self.tracks[j]["L"], 60):
                matched[i] = j
                used.add(j)
        for i, d in enumerate(dets):
            cx, cy, L, S, th = d["rect"]
            if i not in matched:
                self.tracks.append({"id": self.next_id, "cx": cx, "cy": cy, "L": L, "S": S, "th": th,
                                    "hits": 1, "last": now, "box": d["box"]})
                self.next_id += 1
                continue
            t = self.tracks[matched[i]]
            dt = max(0.0, now - t["last"]) or 1 / 30
            k = 1.0 if np.hypot(cx - t["cx"], cy - t["cy"]) > 0.5 * t["S"] else 1 - np.exp(-dt / TRACK_POS_SMOOTH_S)
            t["cx"] += (cx - t["cx"]) * k
            t["cy"] += (cy - t["cy"]) * k
            t["L"] += (L - t["L"]) * k
            t["S"] += (S - t["S"]) * k
            # orientation: smooth the doubled angle (0 and 180 degrees are the same rectangle)
            a = 1.0 if k == 1.0 else 1 - np.exp(-dt / TRACK_HEADING_SMOOTH_S)
            v = np.array([np.cos(np.radians(2 * t["th"])), np.sin(np.radians(2 * t["th"]))]) * (1 - a) + \
                np.array([np.cos(np.radians(2 * th)), np.sin(np.radians(2 * th))]) * a
            t["th"] = float(np.degrees(np.arctan2(v[1], v[0])) / 2 % 180)
            t["hits"] += 1
            t["last"] = now
            t["box"] = d["box"]
        self.tracks = [t for t in self.tracks
                       if now - t["last"] <= (TRACK_HOLD_S if t["hits"] >= TRACK_CONFIRM_FRAMES else 0.25)]
        return [dict(t, held=t["last"] < now) for t in self.tracks if t["hits"] >= TRACK_CONFIRM_FRAMES]


def block_corners(b):
    """Four corners (camera px) of a tracked block, in order around the rectangle."""
    a = np.radians(b["th"])
    ux, uy = np.cos(a) * b["L"] / 2, np.sin(a) * b["L"] / 2
    vx, vy = -np.sin(a) * b["S"] / 2, np.cos(a) * b["S"] / 2
    cx, cy = b["cx"], b["cy"]
    return [(cx - ux - vx, cy - uy - vy), (cx + ux - vx, cy + uy - vy), (cx + ux + vx, cy + uy + vy), (cx - ux + vx, cy - uy + vy)]


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


def build_payload(blocks, mapper, frame_w, frame_h, chips=()):
    items = []
    # White blocks (walls): center plus the four corners, mapped by the browser with its
    # own calibration (covers perspective and mirroring).
    for b in blocks:
        norm = mapper.normalize(b["cx"], b["cy"], frame_w, frame_h)
        if norm is None:
            continue
        corners = [mapper.raw(x, y, frame_w, frame_h) for x, y in block_corners(b)]
        items.append({
            "id": f'white-{b["id"]}',
            "kind": "block",
            "color": "white",
            "nx": round(norm[0], 4),
            "ny": round(norm[1], 4),
            "corners": [[round(x, 4), round(y, 4)] for x, y in corners],
            "angle": round(b["th"], 1),
            "area": int(b["L"] * b["S"]),
            "held": bool(b["held"]),
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


DRAW_LIMIT = 100000  # px; OpenCV 5 rejects coordinates beyond 32 bits, and nothing that far away is visible anyway
_draw_warned = set()


def _ints(*values):
    """Plain Python ints for OpenCV drawing (OpenCV 5 rejects NumPy scalars and values
    beyond 32 bits as points). Non-finite or absurd values are clamped."""
    out = []
    for v in values:
        f = float(v)
        out.append(0 if not np.isfinite(f) else int(round(min(DRAW_LIMIT, max(-DRAW_LIMIT, f)))))
    return tuple(out)


def _box_ok(box, what):
    """False (and a one-time message) for a box with impossible values, so its source can be found."""
    try:
        ok = len(box) == 4 and all(np.isfinite(float(v)) and abs(float(v)) < DRAW_LIMIT for v in box)
    except (TypeError, ValueError):
        ok = False
    if not ok and what not in _draw_warned:
        _draw_warned.add(what)
        print(f"Let op: kader met onmogelijke waarden overgeslagen in de preview ({what}): {box!r}")
    return ok


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
        cx, cy, r = _ints(c["center"][0], c["center"][1], c["r"])
        col = (40, 40, 230) if c["kind"] == "redirect" else (60, 200, 60)
        cv2.circle(frame, (cx, cy), r + 4, (255, 255, 255) if not c["held"] else (160, 160, 160), 2)
        cv2.circle(frame, (cx, cy), r, col, 3)
        label = "splits" if c["kind"] == "split" else "stroom"
        if c["heading"] is not None:
            a = np.radians(c["heading"])
            tip = _ints(cx + np.cos(a) * (r + 40), cy + np.sin(a) * (r + 40))
            cv2.arrowedLine(frame, (cx, cy), tip, (0, 220, 255), 3, tipLength=0.25)
            label += f' {c["heading"]:.0f} deg'
        if c["held"]:
            label += " (vastgehouden)"
        cv2.putText(frame, f'#{c["id"]} {label}', (cx - r, max(20, cy - r - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


def draw_blocks(frame, blocks):
    for b in blocks:
        pts = np.array([_ints(x, y) for x, y in block_corners(b)], np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 255, 0) if not b["held"] else (160, 160, 160), 3)
        x, y = _ints(b["cx"], b["cy"])
        cv2.putText(frame, f'#{b["id"]} blok{" (vastgehouden)" if b["held"] else ""}', (x - 40, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_preview(frame, objects, payload, mapper, fps, n_clients, rejected=()):
    for r in rejected:
        if not _box_ok(r["box"], r.get("reason", "?")):
            continue
        x, y, w, h = _ints(*r["box"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f'genegeerd: {r["reason"]}', (x + 6, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if mapper.corners is not None:
        pts = np.array(mapper.corners, np.int32).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (255, 200, 0), 2)
    for obj in objects:  # raw white detections (tracked blocks are drawn by draw_blocks)
        if not _box_ok(obj["box"], "wit object"):
            continue
        x, y, w, h = _ints(*obj["box"])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 160, 0), 1)
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
# Exposure settings: load / save / keys
# ============================================================
def load_exposure(args):
    st = {"auto": EXPOSURE_AUTO, "exposure": EXPOSURE, "gain": GAIN, "brightness": BRIGHTNESS}
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        for k in st:
            if k in saved and (saved[k] is None or isinstance(saved[k], (bool, int, float))):
                st[k] = saved[k]
        print(f"Belichting geladen uit {os.path.basename(SETTINGS_FILE)}.")
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as e:
        print(f"Let op: {os.path.basename(SETTINGS_FILE)} niet leesbaar ({e}); standaardwaarden gebruikt.")
    if args.exposure is not None:
        st["exposure"], st["auto"] = args.exposure, False
    if args.auto_exposure:
        st["auto"] = True
    if args.gain is not None:
        st["gain"] = args.gain
    if args.brightness is not None:
        st["brightness"] = args.brightness
    st["brightness"] = float(min(3.0, max(1.0, st["brightness"] or 1.0)))
    return st


def save_exposure(st):
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
    except OSError as e:
        print(f"Kon {os.path.basename(SETTINGS_FILE)} niet opslaan: {e}")


def describe_exposure(st, reader):
    if st["auto"]:
        return f'belichting automatisch | helderheid x{st["brightness"]:.1f}'.replace(".", ",")
    secs = 2.0 ** st["exposure"] if st["exposure"] <= 0 else None
    txt = f'belichting {st["exposure"]:g}' + (f" (1/{round(1 / secs)} s)" if secs else "")
    if st["gain"] is not None:
        txt += f' | gain {st["gain"]:g}'
    return txt + f' | helderheid x{st["brightness"]:.1f}'.replace(".", ",")


def handle_exposure_key(key, st, reader):
    """[ or - = darker, ] or + = lighter (exposure, x2 per step), , and . = fine brightness (10%),
    A = auto on/off, S = driver settings window."""
    changed = False
    if key in (ord("["), ord("-"), ord("]"), ord("="), ord("+")):
        step = -1 if key in (ord("["), ord("-")) else 1
        st["auto"] = False
        st["exposure"] = max(-13, min(0, round(st["exposure"]) + step))
        reader.request(auto=False, exposure=st["exposure"])
        changed = True
    elif key in (ord(","), ord(".")):
        st["brightness"] = round(min(3.0, max(1.0, st["brightness"] + (0.1 if key == ord(".") else -0.1))), 1)
        changed = True
    elif key in (ord("a"), ord("A")):
        st["auto"] = not st["auto"]
        reader.request(auto=st["auto"], **({} if st["auto"] else {"exposure": st["exposure"]}))
        changed = True
    elif key in (ord("s"), ord("S")):
        reader.request(dialog=True)
        print("Instellingenvenster van de webcam geopend (alleen beschikbaar via DirectShow op Windows).")
    if changed:
        save_exposure(st)
        print(f"{describe_exposure(st, reader)}  (opgeslagen)")
    return changed


# ============================================================
# Main loop
# ============================================================
async def main(args):
    print(f"OpenCV {cv2.__version__} | websockets {websockets.__version__} | Python {sys.version.split()[0]}")

    reader = CameraReader(args.camera, FRAME_WIDTH, FRAME_HEIGHT)
    exposure = load_exposure(args)
    reader.request(auto=exposure["auto"], gain=exposure["gain"],
                   **({} if exposure["auto"] else {"exposure": exposure["exposure"]}),
                   **({"dialog": True} if args.camera_settings else {}))
    print(f"Camera {args.camera} openen...")
    if not reader.start():
        print("FOUT:", reader.error)
        reader.stop()
        return 1
    print(f"Camera actief via {reader.backend}, resolutie {reader.size[0]}x{reader.size[1]}.")
    acc = reader.controls.get("accepted", {})
    print(f"Webcam: {describe_exposure(exposure, reader)}; "
          f"gemeld door de webcam: exposure={reader.controls.get('exposure')}, gain={reader.controls.get('gain')}.")
    if not exposure["auto"] and acc.get("exposure") is False:
        print("  Let op: de webcam weigert handmatige belichting via OpenCV. Druk S in het cameravenster voor het\n"
              "  instellingenvenster van de webcam, of verlaag de helderheid van de monitor.")
    print("Belichting: [ / ] = grote stap donkerder/lichter, , / . = fijn (10%), A = automatisch aan/uit, S = webcaminstellingen.\n"
          "F = camerabeeld opslaan (onbewerkt, voor het afstellen).")
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
    block_tracker = BlockTracker()
    detect_scale, last_chip_t = 1.0, 0.0
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
                if preview:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    handle_exposure_key(key, exposure, reader)
                continue  # no new frame yet
            seq = new_seq
            if args.mirror and args.flip_y:
                frame = cv2.flip(frame, -1)
            elif args.mirror:
                frame = cv2.flip(frame, 1)
            elif args.flip_y:
                frame = cv2.flip(frame, 0)

            # overexposure as delivered by the webcam (before the software brightness, which cannot lose color)
            clipped = np.count_nonzero(frame.max(axis=2) >= 250) / (frame.shape[0] * frame.shape[1])
            sensor = frame  # chips are detected on the unprocessed image: the brightness step must not clip their colors
            if exposure["brightness"] != 1.0:
                frame = cv2.convertScaleAbs(frame, alpha=exposure["brightness"])  # fine brightness adjustment
            frame_h, frame_w = frame.shape[:2]
            rejected = []
            hsv = cv2.cvtColor(sensor, cv2.COLOR_BGR2HSV)
            _last_hsv["img"] = hsv  # click in the camera window: HSV as the chip detection sees it
            objects, mask = detect_objects(frame, rejected)
            chips, chip_masks = [], None
            if CHIP_DETECTION:
                found, chip_masks = detect_chips(hsv, rejected, sensor, detect_scale)
                now_t = time.perf_counter()
                chips = tracker.update(found, now_t)
                live = [c for c in chips if not c["held"]]
                if live:  # smallest chip in view decides how much the image can be reduced
                    last_chip_t = now_t
                    detect_scale = float(np.clip(DETECT_RADIUS_PX / min(c["r"] for c in live), DETECT_SCALE_MIN, 1.0))
                elif now_t - last_chip_t > 1.0:
                    detect_scale = 1.0  # nothing in view: full resolution, so small chips are found too
            # a white area inside a chip is glare on that chip, not a block
            objects = [o for o in objects if not any(
                np.hypot(o["center"][0] - c["center"][0], o["center"][1] - c["center"][1]) < c["r"] * 1.2 for c in chips)]
            blocks = block_tracker.update(objects, time.perf_counter())
            payload = build_payload(blocks, mapper, frame_w, frame_h, chips)

            if args.debug and time.perf_counter() - last_debug >= 0.5:
                last_debug = time.perf_counter()
                for b in blocks:
                    print(f'[debug] blok #{b["id"]}: pixel=({b["cx"]:.0f},{b["cy"]:.0f}) {b["L"]:.0f}x{b["S"]:.0f}px'
                          f' hoek={b["th"]:.0f} gr{" (vastgehouden)" if b["held"] else ""}')
                if not blocks and not chips:
                    print(f'[debug] geen blok of fiche | genegeerd={len(rejected)}')
                print(f'[debug] {describe_exposure(exposure, reader)} | overbelicht {clipped:.0%} | detectieschaal {detect_scale:.2f}')
                chip_reasons = sorted({r["reason"] for r in rejected if r["reason"].startswith(("rood", "groen"))})
                if chip_reasons and not chips:
                    print("[debug] afgewezen fiche-kandidaten: " + "; ".join(chip_reasons[:5]))
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
                try:
                    draw_preview(frame, objects, payload, mapper, fps, len(clients), rejected)
                    draw_blocks(frame, blocks)
                    draw_chips(frame, chips)
                except (cv2.error, ValueError, OverflowError, TypeError) as e:
                    # the preview is only a debugging aid: detection and the game must keep running
                    if "preview" not in _draw_warned:
                        _draw_warned.add("preview")
                        print(f"Let op: tekenen in het cameravenster mislukt, de herkenning loopt door ({str(e).splitlines()[0]})")
                warn = clipped > CLIPPED_WARN
                cv2.putText(frame, f"{describe_exposure(exposure, reader)} | overbelicht {clipped:.0%}"
                            f"{'  -> druk [ voor minder licht' if warn else ''}",
                            (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if warn else (0, 255, 0), 2)
                cv2.imshow(WINDOW_CAMERA, frame)
                cv2.imshow(WINDOW_MASK, mask_preview(mask, chip_masks))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or window_closed():
                    break
                if key in (ord("f"), ord("F")):  # save the unprocessed camera image (to share when tuning)
                    path = os.path.join(os.path.dirname(SETTINGS_FILE), time.strftime("camerabeeld_%Y%m%d_%H%M%S.png"))
                    print(f"Camerabeeld opgeslagen: {path}" if cv2.imwrite(path, sensor) else "Opslaan mislukt.")
                handle_exposure_key(key, exposure, reader)
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
    p.add_argument("--exposure", type=float, default=None,
                   help="handmatige belichting (Windows: -13 = donker .. 0 = licht, standaard -6)")
    p.add_argument("--brightness", type=float, default=None, help="fijne helderheid in software, 1.0 .. 3.0 (standaard 1.4)")
    p.add_argument("--auto-exposure", action="store_true", help="automatische belichting van de webcam gebruiken")
    p.add_argument("--gain", type=float, default=None, help="versterking van de webcam (0 = geen)")
    p.add_argument("--camera-settings", action="store_true", help="instellingenvenster van de webcam openen (Windows)")
    p.add_argument("--no-chips", action="store_true", help="fiche-detectie uitschakelen (alleen het witte testobject)")
    p.add_argument("--no-preview", action="store_true", default=not SHOW_PREVIEW,
                   help="geen OpenCV-vensters tonen")
    return p.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(parse_args())))
    except KeyboardInterrupt:
        print("Camera-server gestopt.")