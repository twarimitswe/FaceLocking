# src/face_signals.py
"""
Part 2, Chapter 4: smile, blink, and closed-eye detection on the locked
face's region only (never the full frame - see face_tracking.py).

Uses MediaPipe's FaceLandmarker (Tasks API) in IMAGE mode instead of the
legacy mp.solutions.face_mesh.FaceMesh the book lists, since this Python
version's mediapipe build has no mp.solutions (same reason haar_5pt.py
and recognize.py use the Tasks API) - the landmark indices and topology
are identical, so the EAR/smile geometry below is unchanged from the book.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

from .haar_5pt import _DEFAULT_MODEL_PATH

logger = logging.getLogger(__name__)


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14
FACE_LEFT, FACE_RIGHT = 234, 454


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray, idx: Tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (points[i] for i in idx)
    width = max(distance(p1, p4), 1e-6)
    return (distance(p2, p6) + distance(p3, p5)) / (2.0 * width)


@dataclass
class FaceSignals:
    ear: float
    blink: bool
    eyes_closed: bool
    smile_score: float
    smiling: bool


class FaceSignalExtractor:
    """
    Smile and eye-openness are measured relative to each person's own resting
    (neutral) geometry rather than a fixed threshold shared by everyone: a
    fixed ratio that works for one face sits inside another person's normal
    range (see the smile-stuck-on issue this replaced). Both baselines are
    slow EMAs updated only while the signal is NOT currently triggered, so a
    sustained smile or blink can't drag its own baseline toward itself.

    The smile ratio itself is also smoothed (a faster EMA than the
    baseline) before being compared to the thresholds, since single-frame
    landmark jitter - worse in low light - can otherwise bounce a
    borderline ratio back and forth across the on/off boundary and make
    the smile/neutral transition look inconsistent even when the person's
    expression isn't actually changing.
    """

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        blink_min_frames: int = 2,
        blink_max_frames: int = 7,
        closed_frames: int = 8,
        ear_drop_frac: float = 0.22,
        smile_on_margin: float = 0.05,
        smile_off_margin: float = 0.02,
        baseline_alpha: float = 0.03,
        smile_smoothing_alpha: float = 0.5,
    ):
        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames
        self.closed_frames = closed_frames
        self.ear_drop_frac = ear_drop_frac
        self.smile_on_margin = smile_on_margin
        self.smile_off_margin = smile_off_margin
        self.baseline_alpha = baseline_alpha
        self.smile_smoothing_alpha = smile_smoothing_alpha

        self.low_ear_frames = 0
        self.smiling = False
        self.was_eyes_closed = False
        self.baseline_ear: Optional[float] = None
        self.baseline_smile_score: Optional[float] = None
        self.smoothed_smile_score: Optional[float] = None

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def reset(self) -> None:
        self.low_ear_frames = 0
        self.smiling = False
        self.was_eyes_closed = False
        self.baseline_ear = None
        self.baseline_smile_score = None
        self.smoothed_smile_score = None

    def close(self) -> None:
        self.landmarker.close()

    def analyze(self, frame: np.ndarray, bbox) -> Optional[FaceSignals]:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(0.12 * bw), int(0.18 * bh)
        rx1, ry1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        rx2, ry2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks:
            logger.debug("no landmarks found in locked-face ROI (bbox=%s)", bbox)
            return None

        rh, rw = roi.shape[:2]
        lm = result.face_landmarks[0]
        points = np.array(
            [[p.x * rw + rx1, p.y * rh + ry1] for p in lm],
            dtype=np.float32,
        )

        left_ear = eye_aspect_ratio(points, LEFT_EYE)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE)
        ear = 0.5 * (left_ear + right_ear)

        if self.baseline_ear is None:
            self.baseline_ear = ear
        ear_threshold = self.baseline_ear * (1.0 - self.ear_drop_frac)

        blink = False
        blink_held_frames = 0
        if ear < ear_threshold:
            self.low_ear_frames += 1
        else:
            blink_held_frames = self.low_ear_frames
            if self.blink_min_frames <= blink_held_frames <= self.blink_max_frames:
                blink = True
            self.low_ear_frames = 0
            # only drift the "eyes open" baseline while actually open, so a
            # blink or sustained closure can't pull its own trigger point down
            self.baseline_ear += self.baseline_alpha * (ear - self.baseline_ear)
        eyes_closed = self.low_ear_frames >= self.closed_frames

        if blink:
            logger.info("BLINK detected (ear=%.3f, baseline=%.3f, held for %d frames)",
                        ear, self.baseline_ear, blink_held_frames)
        if eyes_closed != self.was_eyes_closed:
            logger.info("eyes %s (ear=%.3f, baseline=%.3f)", "CLOSED" if eyes_closed else "OPEN",
                        ear, self.baseline_ear)
            self.was_eyes_closed = eyes_closed

        face_width = max(distance(points[FACE_LEFT], points[FACE_RIGHT]), 1e-6)
        mouth_width = distance(points[MOUTH_LEFT], points[MOUTH_RIGHT])
        raw_smile_score = mouth_width / face_width

        if self.smoothed_smile_score is None:
            self.smoothed_smile_score = raw_smile_score
        else:
            self.smoothed_smile_score += self.smile_smoothing_alpha * (raw_smile_score - self.smoothed_smile_score)
        smile_score = self.smoothed_smile_score

        logger.debug(
            "smile geometry: face_left=%s face_right=%s mouth_left=%s mouth_right=%s "
            "face_width=%.1f mouth_width=%.1f raw_ratio=%.3f smoothed_ratio=%.3f",
            points[FACE_LEFT], points[FACE_RIGHT], points[MOUTH_LEFT], points[MOUTH_RIGHT],
            face_width, mouth_width, raw_smile_score, smile_score,
        )

        if self.baseline_smile_score is None:
            self.baseline_smile_score = smile_score
        on_threshold = self.baseline_smile_score + self.smile_on_margin
        off_threshold = self.baseline_smile_score + self.smile_off_margin

        was_smiling = self.smiling
        if self.smiling:
            self.smiling = smile_score >= off_threshold
        else:
            self.smiling = smile_score >= on_threshold
            if not self.smiling:
                # only drift the "neutral mouth" baseline while not smiling,
                # so a sustained smile can't pull its own trigger point up
                self.baseline_smile_score += self.baseline_alpha * (smile_score - self.baseline_smile_score)
        if self.smiling != was_smiling:
            logger.info("smile %s (ratio=%.3f, baseline=%.3f)",
                        "ON" if self.smiling else "OFF", smile_score, self.baseline_smile_score)

        logger.debug("signals: ear=%.3f smile_score=%.3f eyes_closed=%s smiling=%s",
                      ear, smile_score, eyes_closed, self.smiling)

        return FaceSignals(
            ear=ear,
            blink=blink,
            eyes_closed=eyes_closed,
            smile_score=smile_score,
            smiling=self.smiling,
        )
