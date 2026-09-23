import logging
import subprocess
from typing import List, Optional

import cv2

logger = logging.getLogger(__name__)


def _is_real_capture_device(index: int) -> Optional[bool]:
    """
    True/False from udev's V4L2 capability flag, or None if it can't be
    determined (non-Linux, udevadm missing, or the device doesn't exist).

    Some cameras expose a second /dev/videoN node alongside their real
    capture node (e.g. for metadata) that still opens and returns a frame
    from cv2.VideoCapture, so a plain open-and-read probe alone picks the
    wrong one; udev's capability flag is what actually distinguishes them.
    """
    try:
        out = subprocess.run(
            ["udevadm", "info", "--query=all", f"--name=/dev/video{index}"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("E: ID_V4L_CAPABILITIES="):
            return "capture" in line
    return None


def _list_capture_devices(max_index: int = 6) -> List[int]:
    """Indices of /dev/videoN nodes that are real capture devices and deliver a frame."""
    found = []
    for i in range(max_index):
        is_capture = _is_real_capture_device(i)
        if is_capture is False:
            continue  # confirmed non-capture (e.g. a metadata-only node); skip without opening it

        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                found.append(i)
        cap.release()
    return found


def pick_camera_index() -> int:
    """
    Prefer a plugged-in USB camera over the laptop's built-in one.

    On Linux, video devices enumerate in attach order: the built-in webcam
    is present at boot and always claims the lowest index(es); a USB
    camera plugged in afterward gets a higher index. So among the working
    capture devices, the highest index is the most recently attached
    camera - the external one if present, the built-in one if it's the
    only camera found.
    """
    devices = _list_capture_devices()
    if not devices:
        raise RuntimeError("No working camera found (checked /dev/video0 through /dev/video5).")
    chosen = max(devices)
    logger.info("camera auto-detect: found %s, using index %d", devices, chosen)
    return chosen


def main():
    cap = cv2.VideoCapture(pick_camera_index())
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try a different index (0/1/2).")
    print("Camera test. Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break
        cv2.imshow("Camera Test", frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
