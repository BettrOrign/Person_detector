"""Capture a single frame from webcam using OpenCV and save to file."""

import sys
import cv2


def main() -> None:
    """Capture one frame from /dev/video0 and write it to the given path."""
    if len(sys.argv) < 2:
        print("Usage: python3 capture.py <output_path>", file=sys.stderr)
        sys.exit(1)

    output_path = sys.argv[1]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("FAIL: Could not open video device", file=sys.stderr)
        sys.exit(1)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("FAIL: Could not read frame", file=sys.stderr)
        sys.exit(1)

    cv2.imwrite(output_path, frame)
    print("OK", file=sys.stderr)


if __name__ == "__main__":
    main()