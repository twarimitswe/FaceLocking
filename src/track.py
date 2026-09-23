# src/track.py
"""
Servo face-following: pick one enrolled identity, and a pan servo on an
ESP32 (see firmware/servo_tracker/servo_tracker.ino) turns to keep that
person's face centered in frame.

Reuses the same detection/recognition stack as recognize.py
(Haar per-face ROI -> MediaPipe FaceLandmarker 5pt -> align -> ArcFace
embedding -> cosine match against data/db/face_db.npz).

Run:
 python -m src.track [--camera N] [--host servotracker.local] [--invert]

Keys:
 q : quit
 c : re-center the servo
"""

from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import requests

from .camera import pick_camera_index
from .recognize import HaarFaceMesh5pt, ArcFaceEmbedderONNX, FaceDBMatcher, load_db_npz
from .haar_5pt import align_face_5pt


# -------------------------
# Servo client (own thread, so a slow/unreachable ESP32 never stalls the camera loop)
# -------------------------

class ServoClient:
    def __init__(self, host: str, invert: bool = False, send_hz: float = 20.0, timeout_s: float = 0.2,
                 min_angle: float = 0.0, max_angle: float = 180.0):
        self.base_url = f"http://{host}"
        self.invert = invert
        self.send_interval = 1.0 / send_hz
        self.timeout_s = timeout_s
        self.min_angle = min_angle
        self.max_angle = max_angle

        self._desired_angle = 90
        self._last_sent_angle: Optional[int] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_angle(self, angle: float) -> None:
        angle = int(round(max(self.min_angle, min(self.max_angle, angle))))
        if self.invert:
            angle = 180 - angle
        with self._lock:
            self._desired_angle = angle

    def center(self) -> None:
        self.set_angle(90)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                angle = self._desired_angle
            if angle != self._last_sent_angle:
                try:
                    requests.get(f"{self.base_url}/servo", params={"angle": angle}, timeout=self.timeout_s)
                    self._last_sent_angle = angle
                except requests.RequestException:
                    pass  # ESP32 unreachable this tick; retry next tick
            time.sleep(self.send_interval)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)


# -------------------------
# Pan controller: face x-position -> servo angle
# -------------------------

def face_center_to_angle(face_cx: float, frame_w: int, gain: float = 30.0) -> float:
    """
    error in [-1, 1] (face left-of-center -> negative) mapped onto a
    90-degree-centered servo sweep.

    gain is kept modest (not the full +/-90 deg a frame-edge error could
    imply) so a correction's angular distance stays within what the servo
    can physically finish moving during one lock-hold window at its slew
    rate - otherwise every correction gets interrupted mid-move by the next
    one and the camera never actually settles.
    """
    error = (face_cx - frame_w / 2.0) / (frame_w / 2.0)
    return 90.0 - error * gain


# -------------------------
# Search sweep: used once the target has been missing for a while, so the
# servo keeps scanning for them instead of just sitting still.
# -------------------------

class SearchSweep:
    def __init__(self, speed_dps: float = 30.0, min_angle: float = 0.0, max_angle: float = 180.0):
        self.speed_dps = speed_dps
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.angle = (min_angle + max_angle) / 2.0
        self.direction = 1

    def sync(self, angle: float) -> None:
        """Call while tracking normally, so a sweep starts from wherever we last were."""
        self.angle = angle

    def step(self, dt: float) -> float:
        self.angle += self.direction * self.speed_dps * dt
        if self.angle >= self.max_angle:
            self.angle = self.max_angle
            self.direction = -1
        elif self.angle <= self.min_angle:
            self.angle = self.min_angle
            self.direction = 1
        return self.angle


# -------------------------
# Main
# -------------------------

def pick_target_name(db_path: Path) -> str:
    db = load_db_npz(db_path)
    names = sorted(db.keys())
    if not names:
        raise RuntimeError(f"No enrolled identities in {db_path}. Run `python -m src.enroll` first.")

    print("Enrolled identities:")
    for i, n in enumerate(names, 1):
        print(f"  {i}. {n}")

    while True:
        choice = input(f"Select who to follow (1-{len(names)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Invalid choice, try again.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=None,
                         help="cv2 camera index; auto-detects a plugged-in USB camera over the "
                              "built-in one when omitted")
    parser.add_argument("--host", default="servotracker.local", help="ESP32 mDNS hostname or IP")
    parser.add_argument("--invert", action="store_true", help="flip pan direction if the servo turns the wrong way")
    parser.add_argument("--min-angle", type=float, default=0.0, help="lowest safe servo angle")
    parser.add_argument("--max-angle", type=float, default=180.0,
                         help="highest safe servo angle, matching SERVO_MAX_ANGLE in the .ino")
    args = parser.parse_args()

    db_path = Path("data/db/face_db.npz")
    target_name = pick_target_name(db_path)

    det = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(model_path="models/embedder_arcface.onnx", input_size=(112, 112), debug=False)
    matcher = FaceDBMatcher(db=load_db_npz(db_path), dist_thresh=0.34)
    servo = ServoClient(host=args.host, invert=args.invert, min_angle=args.min_angle, max_angle=args.max_angle)

    camera_index = args.camera if args.camera is not None else pick_camera_index()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera index {camera_index} not available.")

    print(f"Following '{target_name}'. Servo at http://{args.host}/  Press 'c' to re-center, 'q' to quit.")

    LOST_GRACE_S = 1.0     # only start sweeping after a full second without seeing the target
    LOCK_DEADZONE_DEG = 4  # ignore position noise smaller than this once locked on, so it holds still
    LOCK_HOLD_S = 1.0       # after the servo finishes a correction, wait this long before considering another
    SERVO_SPEED_DPS = 15.0  # must match the firmware's slew rate, to estimate when a move has actually landed
    search = SearchSweep(speed_dps=SERVO_SPEED_DPS, min_angle=args.min_angle, max_angle=args.max_angle)
    last_t = time.time()
    last_seen_t: Optional[float] = None
    locked_angle: Optional[float] = None
    eval_allowed_t: Optional[float] = None
    last_debug_t = 0.0
    was_searching = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            dt = now - last_t
            last_t = now

            H, W = frame.shape[:2]
            faces = det.detect(frame, max_faces=5)
            vis = frame.copy()

            best_name = None
            best_sim = -1.0
            best_face = None
            debug_matches = []

            for f in faces:
                aligned, _ = align_face_5pt(frame, f.kps, out_size=(112, 112))
                emb = embedder.embed(aligned)
                mr = matcher.match(emb)
                debug_matches.append((mr.name, mr.similarity, mr.distance, mr.accepted))

                color = (0, 128, 255)
                label = mr.name if mr.name is not None else "Unknown"
                if mr.accepted and mr.name == target_name and mr.similarity > best_sim:
                    best_sim = mr.similarity
                    best_name = mr.name
                    best_face = f
                if mr.accepted and mr.name == target_name:
                    color = (0, 255, 0)

                cv2.rectangle(vis, (f.x1, f.y1), (f.x2, f.y2), color, 2)
                cv2.putText(vis, label, (f.x1, max(0, f.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if now - last_debug_t > 1.0:
                last_debug_t = now
                print(f"[track] faces={len(faces)} matches={debug_matches} locked={locked_angle}", flush=True)

            if best_face is not None:
                if was_searching:
                    print(f"[track] REACQUIRED {target_name}, stopping search", flush=True)
                    was_searching = False
                last_seen_t = now
                cx = (best_face.x1 + best_face.x2) / 2.0
                raw_angle = max(args.min_angle, min(args.max_angle, face_center_to_angle(cx, W)))

                # Hold still on small position noise. After any correction, wait for the
                # servo to physically finish that move (estimated from its slew rate) PLUS
                # a full second before evaluating another one - since the camera itself
                # sits on the pan servo, reading its position while it's still mid-move
                # would otherwise trigger another correction before the first one lands,
                # so it never actually settles.
                past_hold = eval_allowed_t is None or now >= eval_allowed_t
                if locked_angle is None or (past_hold and abs(raw_angle - locked_angle) > LOCK_DEADZONE_DEG):
                    travel_s = abs(raw_angle - (locked_angle if locked_angle is not None else 90.0)) / SERVO_SPEED_DPS
                    eval_allowed_t = now + travel_s + LOCK_HOLD_S
                    servo.set_angle(raw_angle)
                    locked_angle = raw_angle
                    search.sync(raw_angle)

                cv2.putText(vis, f"locked on {target_name} | angle~{locked_angle:.0f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                lost_for = now - last_seen_t if last_seen_t is not None else float("inf")
                if lost_for <= LOST_GRACE_S:
                    cv2.putText(vis, f"waiting for {target_name}...", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    if not was_searching:
                        print(f"[track] search STARTED after {lost_for:.2f}s without {target_name}", flush=True)
                        was_searching = True
                    angle = search.step(dt)
                    servo.set_angle(angle)
                    locked_angle = None
                    eval_allowed_t = None
                    cv2.putText(vis, f"searching for {target_name} | angle~{angle:.0f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            cv2.line(vis, (W // 2, 0), (W // 2, H), (255, 255, 255), 1)
            cv2.imshow("track", vis)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                servo.center()
    finally:
        servo.close()
        det.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
