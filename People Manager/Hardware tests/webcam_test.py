
import cv2
import numpy as np

# Detect a white AirPods case using brightness, size and shape.
# Best results: closed case on a dark, matte surface.
# Press Q or ESC to quit.

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# White object detection
BRIGHTNESS_MIN = 200
SATURATION_MAX = 145

# Ignore small objects
MIN_AREA = 3000
MAX_AREA = 150000

# Allow different case orientations
MIN_ASPECT_RATIO = 0.65
MAX_ASPECT_RATIO = 3.0

# Shape filters
MIN_EXTENT = 0.30
MIN_SOLIDITY = 0.70

# Image processing
OPEN_KERNEL_SIZE = 3
CLOSE_KERNEL_SIZE = 7


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

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

    print("Place the AirPods case on a dark, matte surface.")
    print("Keep other white objects out of the camera view.")
    print("Press Q or ESC to quit.")

    while True:
        ok, frame = cap.read()

        if not ok:
            print("Could not read webcam frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Detect bright, low-saturation regions
        lower = np.array(
            [0, 0, BRIGHTNESS_MIN], dtype=np.uint8
        )
        upper = np.array(
            [180, SATURATION_MAX, 255], dtype=np.uint8
        )

        mask = cv2.inRange(hsv, lower, upper)

        # Remove noise and fill small gaps
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

        display = frame.copy()
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by actual contour area
            if area < MIN_AREA or area > MAX_AREA:
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

            # Show possible objects in yellow for debugging
            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                1
            )

            # Filter by shape
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

            # Prefer a compact, case-like shape
            aspect_score = max(
                0.0,
                1.0 - abs(np.log(aspect_ratio / 1.6)) / 1.5
            )

            score = (
                0.40 * aspect_score
                + 0.30 * extent
                + 0.30 * solidity
            )

            candidates.append((
                area, score, x, y, w, h,
                aspect_ratio, extent, solidity
            ))

        if candidates:
            # Select the largest suitable object
            candidates.sort(
                key=lambda c: c[0],
                reverse=True
            )

            (
                area, score, x, y, w, h,
                aspect_ratio, extent, solidity
            ) = candidates[0]

            cx = x + w // 2
            cy = y + h // 2

            # Draw detected case
            cv2.rectangle(
                display,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                3
            )

            cv2.circle(
                display,
                (cx, cy),
                6,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                "LARGEST OBJECT CANDIDATE",
                (x, max(25, y - 35)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            details = (
                f"Area={area:.0f} "
                f"AR={aspect_ratio:.2f} "
                f"Extent={extent:.2f} "
                f"Solidity={solidity:.2f}"
            )

            cv2.putText(
                display,
                details,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            status = (
                f"OBJECT DETECTED | Center: ({cx}, {cy})"
            )
            status_colour = (0, 255, 0)

        else:
            status = "NO SUITABLE OBJECT"
            status_colour = (0, 0, 255)

        cv2.putText(
            display,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            status_colour,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            "Q / ESC = quit",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.imshow("AirPods Case Detection", display)
        cv2.imshow("White Object Mask", mask)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
