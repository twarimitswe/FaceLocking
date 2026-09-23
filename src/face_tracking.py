# src/face_tracking.py
"""
Part 2: identity lock, tracking, and position output for one enrolled
target. Reuses Part 1's detection/recognition stack (src/recognize.py)
and alignment (src/align.py) rather than duplicating them.

Run:
 python -m src.face_tracking --target "SomeEnrolledName"

Keys:
 q : quit
"""

import argparse
import logging
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .align import align_face_5pt
from .face_signals import FaceSignalExtractor
from .recognize import (
    ArcFaceEmbedderONNX,
    FaceDBMatcher,
    HaarFaceMesh5pt,
    load_db_npz,
)

logger = logging.getLogger(__name__)


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class TrackingSignal:
    error_x: float
    error_y: float
    horizontal: str
    vertical: str


class LockedFaceTracker:
    def __init__(
        self,
        target_name: str,
        detector,
        embedder,
        matcher,
        verify_every: int = 10,
        lost_timeout: int = 24,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone
        self.state = LockState.SEARCHING
        self.last_box = None
        self.smooth_center = None
        self.lost_frames = 0
        self.frame_index = 0

    @staticmethod
    def box(face):
        return (face.x1, face.y1, face.x2, face.y2)

    def identity(self, frame, face):
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        return self.matcher.match(self.embedder.embed(aligned))

    def target_is_verified(self, frame, face) -> bool:
        match = self.identity(frame, face)
        return match.accepted and match.name == self.target_name

    def acquire(self, frame, faces):
        best = None
        best_similarity = -1.0
        for face in faces:
            match = self.identity(frame, face)
            logger.debug(
                "acquire: candidate box=%s matched=%r similarity=%.3f distance=%.3f accepted=%s",
                self.box(face), match.name, match.similarity, match.distance, match.accepted,
            )
            if (
                match.accepted
                and match.name == self.target_name
                and match.similarity > best_similarity
            ):
                best, best_similarity = face, match.similarity
        if best is not None:
            logger.info(
                "acquired target %r at box=%s similarity=%.3f",
                self.target_name, self.box(best), best_similarity,
            )
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None

        # with only one face in view there's nothing to disambiguate against -
        # a fast-moving target can outrun the displacement penalty below and
        # get wrongly rejected even though it's obviously still the same
        # (only) face, so skip straight to it and let periodic ArcFace
        # re-verification catch a genuine handoff to someone else
        if len(faces) == 1:
            logger.debug("associate: single face in view, skipping displacement scoring")
            return faces[0]

        last_center = center(self.last_box)
        last_diag = max(np.linalg.norm(
            np.array([self.last_box[2] - self.last_box[0],
                      self.last_box[3] - self.last_box[1]], dtype=np.float32)
        ), 1.0)
        ranked = []
        for face in faces:
            box = self.box(face)
            overlap = iou(self.last_box, box)
            displacement = np.linalg.norm(center(box) - last_center) / last_diag
            score = overlap - 0.20 * displacement
            ranked.append((score, face))
        score, candidate = max(ranked, key=lambda item: item[0])
        chosen = candidate if score > -0.45 else None
        logger.debug(
            "associate: best_score=%.3f box=%s -> %s",
            score, self.box(candidate), "kept" if chosen is not None else "rejected",
        )
        return chosen

    def update(self, frame):
        self.frame_index += 1
        faces = self.detector.detect(frame, max_faces=8)
        logger.debug("frame=%d state=%s faces_detected=%d", self.frame_index, self.state.name, len(faces))

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)
            due_for_verify = self.state == LockState.LOST or self.frame_index % self.verify_every == 0
            if candidate is not None and due_for_verify:
                verified = self.target_is_verified(frame, candidate)
                logger.debug("periodic re-verification: verified=%s", verified)
                if not verified:
                    candidate = None

        if candidate is None:
            self.lost_frames += 1
            if self.last_box is not None and self.state != LockState.LOST:
                logger.info("target %r LOST (frame=%d, was at %s)", self.target_name, self.frame_index, self.last_box)
            if self.last_box is not None:
                self.state = LockState.LOST
            if self.lost_frames > self.lost_timeout:
                if self.state == LockState.LOST:
                    logger.info(
                        "target %r missing for %d frames (> lost_timeout=%d) -> back to SEARCHING",
                        self.target_name, self.lost_frames, self.lost_timeout,
                    )
                self.state = LockState.SEARCHING
                self.last_box = None
                self.smooth_center = None
            return None, None

        if self.state != LockState.LOCKED:
            logger.info("target %r LOCKED (frame=%d)", self.target_name, self.frame_index)
        self.state = LockState.LOCKED
        self.lost_frames = 0
        self.last_box = self.box(candidate)
        raw_center = center(self.last_box)

        if self.smooth_center is None:
            self.smooth_center = raw_center
        else:
            a = self.ema_alpha
            self.smooth_center = a * raw_center + (1.0 - a) * self.smooth_center
        signal = self.position_signal(frame.shape)
        logger.debug(
            "position: raw_center=%s smoothed=%s error=(%.3f,%.3f) H=%s V=%s",
            raw_center, self.smooth_center, signal.error_x, signal.error_y,
            signal.horizontal, signal.vertical,
        )
        return candidate, signal

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        ex = float((self.smooth_center[0] - width / 2.0) / (width / 2.0))
        ey = float((self.smooth_center[1] - height / 2.0) / (height / 2.0))
        horizontal = "CENTER"
        vertical = "CENTER"
        if ex < -self.dead_zone:
            horizontal = "LEFT"
        elif ex > self.dead_zone:
            horizontal = "RIGHT"
        if ey < -self.dead_zone:
            vertical = "UP"
        elif ey > self.dead_zone:
            vertical = "DOWN"
        return TrackingSignal(ex, ey, horizontal, vertical)


def draw_label(frame, text, xy, color, scale=0.62):
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="enrolled identity to lock")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.34)
    parser.add_argument("--debug", action="store_true",
                         help="verbose per-frame logs: detection, position, lock state, smile/blink")
    parser.add_argument("--ear-drop-frac", type=float, default=0.22,
                         help="fraction below each person's own open-eye EAR baseline that counts as closed")
    parser.add_argument("--smile-on-margin", type=float, default=0.05,
                         help="how far the mouth/face ratio must rise above each person's own neutral "
                              "baseline to count as smiling")
    parser.add_argument("--smile-off-margin", type=float, default=0.02,
                         help="how close the ratio must return to baseline to clear smiling (must be "
                              "smaller than --smile-on-margin)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    detector = HaarFaceMesh5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    matcher = FaceDBMatcher(
        load_db_npz(Path("data/db/face_db.npz")),
        dist_thresh=args.threshold,
    )
    tracker = LockedFaceTracker(args.target, detector, embedder, matcher)
    signals = FaceSignalExtractor(
        ear_drop_frac=args.ear_drop_frac,
        smile_on_margin=args.smile_on_margin,
        smile_off_margin=args.smile_off_margin,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Camera not available")

    logger.info(
        "face_tracking started: target=%r camera=%d threshold=%.2f",
        args.target, args.camera, args.threshold,
    )

    blink_total = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            locked_face, position = tracker.update(frame)
            view = frame.copy()

            state_text = f"{tracker.state.name}: {args.target}"
            state_color = (0, 180, 0) if locked_face is not None else (0, 140, 255)
            draw_label(view, state_text, (12, 28), state_color, 0.72)

            if locked_face is not None:
                box = tracker.box(locked_face)
                x1, y1, x2, y2 = box
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 170, 0), 3)
                face_state = signals.analyze(frame, box)
                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1
                        logger.debug("blink_total=%d", blink_total)

                    expression = "SMILE" if face_state.smiling else "NEUTRAL"
                    eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"
                    draw_label(view, expression, (x1, max(55, y1 - 50)), (0, 255, 255))
                    draw_label(view, f"{eye_text} blinks={blink_total}",
                               (x1, max(78, y1 - 25)), (255, 255, 0))
                    draw_label(view, f"EAR={face_state.ear:.3f} "
                               f"smile={face_state.smile_score:.3f}",
                               (12, view.shape[0] - 18), (255, 255, 255), 0.52)
                draw_label(view,
                           f"H={position.horizontal} V={position.vertical} "
                           f"error=({position.error_x:+.2f},{position.error_y:+.2f})",
                           (12, 56), (255, 170, 0), 0.60)
            else:
                signals.reset()

            h, w = view.shape[:2]
            dz = tracker.dead_zone
            cv2.rectangle(view,
                          (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                          (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                          (120, 120, 120), 1)
            cv2.imshow("Locked Face Tracking", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        signals.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
