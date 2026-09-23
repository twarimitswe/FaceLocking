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
    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        ear_threshold: float = 0.21,
        blink_min_frames: int = 2,
        blink_max_frames: int = 7,
        closed_frames: int = 8,
        smile_on: float = 0.38,
        smile_off: float = 0.35,
    ):
        self.ear_threshold = ear_threshold
        self.blink_min_frames = blink_min_frames
        self.blink_max_frames = blink_max_frames
        self.closed_frames = closed_frames
        self.smile_on = smile_on
        self.smile_off = smile_off
        self.low_ear_frames = 0
        self.smiling = False
        self.was_eyes_closed = False

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

        blink = False
        blink_held_frames = 0
        if ear < self.ear_threshold:
            self.low_ear_frames += 1
        else:
            blink_held_frames = self.low_ear_frames
            if self.blink_min_frames <= blink_held_frames <= self.blink_max_frames:
                blink = True
            self.low_ear_frames = 0
        eyes_closed = self.low_ear_frames >= self.closed_frames

        if blink:
            logger.info("BLINK detected (ear=%.3f, held for %d frames)", ear, blink_held_frames)
        if eyes_closed != self.was_eyes_closed:
            logger.info("eyes %s (ear=%.3f)", "CLOSED" if eyes_closed else "OPEN", ear)
            self.was_eyes_closed = eyes_closed

        face_width = max(distance(points[FACE_LEFT], points[FACE_RIGHT]), 1e-6)
        mouth_width = distance(points[MOUTH_LEFT], points[MOUTH_RIGHT])
        smile_score = mouth_width / face_width
        logger.debug(
            "smile geometry: face_left=%s face_right=%s mouth_left=%s mouth_right=%s "
            "face_width=%.1f mouth_width=%.1f ratio=%.3f",
            points[FACE_LEFT], points[FACE_RIGHT], points[MOUTH_LEFT], points[MOUTH_RIGHT],
            face_width, mouth_width, smile_score,
        )
        was_smiling = self.smiling
        if self.smiling:
            self.smiling = smile_score >= self.smile_off
        else:
            self.smiling = smile_score >= self.smile_on
        if self.smiling != was_smiling:
            logger.info("smile %s (score=%.3f)", "ON" if self.smiling else "OFF", smile_score)

        logger.debug("signals: ear=%.3f smile_score=%.3f eyes_closed=%s smiling=%s",
                      ear, smile_score, eyes_closed, self.smiling)

        return FaceSignals(
            ear=ear,
            blink=blink,
            eyes_closed=eyes_closed,
            smile_score=smile_score,
            smiling=self.smiling,
        )
