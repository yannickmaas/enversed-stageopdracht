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

    def normalize(self, cx, cy, frame_w, frame_h):
        """Returns (nx, ny), or None if the point lies outside the play area."""
        if self.matrix is None:
            nx, ny = cx / frame_w, cy / frame_h
        else:
            pt = np.array([[[cx, cy]]], np.float32)
            nx, ny = cv2.perspectiveTransform(pt, self.matrix)[0][0]
        margin = 0.02
        if not (-margin <= nx <= 1 + margin and -margin <= ny <= 1 + margin):
            return None
        return min(1.0, max(0.0, float(nx))), min(1.0, max(0.0, float(ny)))


def build_payload(objects, mapper, frame_w, frame_h):
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
        })
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
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Klik in camerabeeld: ({x}, {y})")


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
            objects, mask = detect_objects(frame, rejected)
            payload = build_payload(objects, mapper, frame_w, frame_h)

            if args.debug and time.perf_counter() - last_debug >= 0.5:
                last_debug = time.perf_counter()
                if payload["detected"]:
                    o = objects[0]
                    print(f'[debug] pixel=({o["center"][0]:.0f},{o["center"][1]:.0f}) van {frame_w}x{frame_h}'
                          f' -> nx={payload["nx"]:.3f} ny={payload["ny"]:.3f}'
                          f' | kandidaten={len(objects)} genegeerd={len(rejected)}')
                else:
                    print(f'[debug] geen object | genegeerd={len(rejected)}')

            now = time.perf_counter()
            # Send state changes (object / no object) immediately, otherwise rate-limited.
            if payload["detected"] != last_detected or now - last_send >= send_interval:
                await broadcast(payload)
                last_send = now
                last_detected = payload["detected"]

            fps_n += 1
            if now - fps_t >= 1.0:
                fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0

            if preview:
                draw_preview(frame, objects, payload, mapper, fps, len(clients), rejected)
                cv2.imshow(WINDOW_CAMERA, frame)
                cv2.imshow(WINDOW_MASK, mask)
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
    p.add_argument("--no-preview", action="store_true", default=not SHOW_PREVIEW,
                   help="geen OpenCV-vensters tonen")
    return p.parse_args()


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main(parse_args())))
    except KeyboardInterrupt:
        print("Camera-server gestopt.")
