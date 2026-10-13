
import cv2
import numpy as np
import math

# ==========================================
# CAMERA SETTINGS
# ==========================================

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ==========================================
# EXISTING WORKING DETECTION SETTINGS
# ==========================================

BRIGHTNESS_MIN = 200
SATURATION_MAX = 145

MIN_AREA = 3000
MAX_AREA = 150000

MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 3.0

MIN_EXTENT = 0.30
MIN_SOLIDITY = 0.70

OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 7

# ==========================================
# TRACKING SETTINGS
# ==========================================

SMOOTHING = 0.35

# Maximum frames to keep tracking after losing the case
MAX_MISSED_FRAMES = 12

# Search radius around the previous case position.
# Increase this if you move the case quickly.
MAX_TRACK_DISTANCE = 150

# Weight of position, size and shape in candidate selection
POSITION_WEIGHT = 0.60
SIZE_WEIGHT = 0.20
SHAPE_WEIGHT = 0.20

# ==========================================
# DETECTION
# ==========================================

def detect_candidates(frame, kernel_open, kernel_close):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array(
        [0, 0, BRIGHTNESS_MIN], dtype=np.uint8
    )
    upper = np.array(
        [180, SATURATION_MAX, 255], dtype=np.uint8
    )

    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel_open
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel_close
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

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

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / float(hull_area)

        if not (
            MIN_ASPECT_RATIO
            <= aspect_ratio
            <= MAX_ASPECT_RATIO
        ):
            continue

        if extent < MIN_EXTENT:
            continue

        if solidity < MIN_SOLIDITY:
            continue

        center = (x + w // 2, y + h // 2)

        # Compact, case-like shape score
        aspect_score = max(
            0.0,
            1.0 - abs(math.log(aspect_ratio / 1.6)) / 1.5
        )

        shape_score = (
            0.40 * aspect_score
            + 0.30 * extent
            + 0.30 * solidity
        )

        candidates.append({
            "center": center,
            "box": (x, y, w, h),
            "area": area,
            "shape_score": shape_score,
            "aspect_ratio": aspect_ratio
        })

    return candidates, mask


# ==========================================
# TRACKING
# ==========================================

def choose_candidate(candidates, previous_center, previous_area):
    if not candidates:
        return None

    # Initial acquisition: use the largest suitable candidate.
    # Once acquired, prioritize continuity with the tracked case.
    if previous_center is None:
        return max(candidates, key=lambda c: c["area"])

    scored_candidates = []

    for candidate in candidates:
        cx, cy = candidate["center"]

        distance = math.hypot(
            cx - previous_center[0],
            cy - previous_center[1]
        )

        # Reject objects too far from the last known position.
        if distance > MAX_TRACK_DISTANCE:
            continue

        position_score = max(
            0.0, 1.0 - distance / MAX_TRACK_DISTANCE
        )

        # Compare current size to the previously tracked size.
        size_ratio = candidate["area"] / max(previous_area, 1)

        size_score = math.exp(
            -abs(math.log(max(size_ratio, 0.01)))
        )

        shape_score = candidate["shape_score"]

        total_score = (
            POSITION_WEIGHT * position_score
            + SIZE_WEIGHT * size_score
            + SHAPE_WEIGHT * shape_score
        )

        scored_candidates.append(
            (total_score, candidate)
        )

    if not scored_candidates:
        return None

    scored_candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return scored_candidates[0][1]


# ==========================================
# MAIN
# ==========================================

def main():
    cap = cv2.VideoCapture(
        CAMERA_INDEX, cv2.CAP_DSHOW
    )

    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Could not open camera. Try CAMERA_INDEX = 1.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    kernel_open = np.ones(
        (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE), np.uint8
    )
    kernel_close = np.ones(
        (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE), np.uint8
    )

    tracked_position = None
    tracked_area = None
    missed_frames = 0

    trail = []

    print("AirPods tracking started.")
    print("Hold and move the case slowly.")
    print("Press Q or ESC to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read webcam frame.")
            break

        candidates, mask = detect_candidates(
            frame, kernel_open, kernel_close
        )

        display = frame.copy()

        # Show all valid candidates in yellow.
        for candidate in candidates:
            x, y, w, h = candidate["box"]

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                1
            )

        selected = choose_candidate(
            candidates,
            tracked_position,
            tracked_area if tracked_area is not None else 1
        )

        if selected is not None:
            raw_x, raw_y = selected["center"]

            if tracked_position is None:
                tracked_position = (
                    float(raw_x), float(raw_y)
                )
            else:
                old_x, old_y = tracked_position

                tracked_position = (
                    old_x + SMOOTHING * (raw_x - old_x),
                    old_y + SMOOTHING * (raw_y - old_y)
                )

            # Update the size estimate gradually to avoid
            # abrupt changes when the contour varies slightly.
            if tracked_area is None:
                tracked_area = float(selected["area"])
            else:
                tracked_area = (
                    0.8 * tracked_area
                    + 0.2 * selected["area"]
                )

            missed_frames = 0

            tx, ty = (
                int(tracked_position[0]),
                int(tracked_position[1])
            )

            x, y, w, h = selected["box"]

            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.circle(
                display,
                (tx, ty),
                7,
                (0, 0, 255),
                -1
            )

            cv2.drawMarker(
                display,
                (tx, ty),
                (255, 0, 0),
                cv2.MARKER_CROSS,
                24,
                2
            )

            trail.append((tx, ty))

            if len(trail) > 40:
                trail.pop(0)

            status = "TRACKING CASE"
            status_color = (0, 255, 0)

            cv2.putText(
                display,
                f"Position: X={tx} Y={ty}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                f"Area: {selected['area']:.0f}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        else:
            missed_frames += 1

            if missed_frames <= MAX_MISSED_FRAMES:
                status = "CASE OCCLUDED / SEARCHING"
                status_color = (0, 165, 255)

                # Keep the last known position temporarily.
                if tracked_position is not None:
                    tx, ty = (
                        int(tracked_position[0]),
                        int(tracked_position[1])
                    )

                    cv2.circle(
                        display,
                        (tx, ty),
                        8,
                        (0, 165, 255),
                        2
                    )
            else:
                status = "TRACKING LOST - REACQUIRING"
                status_color = (0, 0, 255)

                tracked_position = None
                tracked_area = None
                trail.clear()

        # Draw the path of the tracked case.
        for i in range(1, len(trail)):
            cv2.line(
                display,
                trail[i - 1],
                trail[i],
                (255, 100, 0),
                2
            )

        cv2.putText(
            display,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            "Yellow = candidates | Green = selected",
            (20, display.shape[0] - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            "Q / ESC = quit",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("AirPods Case Tracking", display)
        cv2.imshow("Detection Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
