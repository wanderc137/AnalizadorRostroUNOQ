"""Detección de rostros (Haar cascade) y dibujo de overlay.

La detección se ejecuta cada ``INFER_EVERY_N`` frames en el pipeline; este
módulo simplemente recibe un frame, devuelve las caras con su predicción y
sabe pintar cajas/etiquetas sobre una copia del frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.models_loader import FacePrediction, ModelHandle

DEFAULT_CASCADE_NAME = "haarcascade_frontalface_default.xml"
BOX_COLOR = (0, 255, 180)
BOX_COLOR_MUTED = (0, 190, 255)
TEXT_COLOR = (10, 12, 16)
NO_MODEL_LABEL = "Rostro detectado | modelo no cargado"


@dataclass
class FaceResult:
    """Rostro detectado + predicción opcional."""

    box: Tuple[int, int, int, int]
    prediction: Optional[FacePrediction] = None

    @property
    def confidence(self) -> float:
        return self.prediction.confidence if self.prediction is not None else 0.0


def _resolve_cascade_path(haar_path: Optional[Path]) -> str:
    if haar_path is not None:
        candidate = Path(haar_path)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"HAAR_PATH no existe: {candidate}")
    try:
        base = cv2.data.haarcascades  # type: ignore[attr-defined]
    except AttributeError as exc:  # pragma: no cover - build sin cv2.data
        raise FileNotFoundError(
            "opencv-python no incluye cv2.data; define HAAR_PATH."
        ) from exc
    return str(Path(base) / DEFAULT_CASCADE_NAME)


class FaceDetector:
    """Detector Haar frontalface con clasificación opcional."""

    def __init__(
        self,
        model: ModelHandle,
        haar_path: Optional[Path] = None,
        conf_threshold: float = 0.6,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: Tuple[int, int] = (60, 60),
    ) -> None:
        cascade_file = _resolve_cascade_path(haar_path)
        self._cascade = cv2.CascadeClassifier(cascade_file)
        if self._cascade.empty():
            raise RuntimeError(
                f"No se pudo cargar el Haar cascade: {cascade_file}"
            )
        self.cascade_path = cascade_file
        self.model = model
        self.conf_threshold = float(conf_threshold)
        self.scale_factor = float(scale_factor)
        self.min_neighbors = int(min_neighbors)
        self.min_size = (int(min_size[0]), int(min_size[1]))

    # ------------------------------------------------------------ detección

    @staticmethod
    def _crop_with_margin(
        frame: np.ndarray, box: Tuple[int, int, int, int], margin: float = 0.06
    ) -> Optional[np.ndarray]:
        x, y, w, h = box
        pad_x = int(w * margin)
        pad_y = int(h * margin)
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(frame.shape[1], x + w + pad_x)
        y1 = min(frame.shape[0], y + h + pad_y)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return frame[y0:y1, x0:x1]

    def detect(self, frame: np.ndarray) -> List[FaceResult]:
        """Detecta rostros y clasifica cada ROI (si hay modelo cargado)."""
        if frame is None or frame.size == 0:
            return []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
        )

        results: List[FaceResult] = []
        for raw_box in boxes:
            box = (int(raw_box[0]), int(raw_box[1]), int(raw_box[2]), int(raw_box[3]))
            prediction: Optional[FacePrediction] = None
            if self.model.loaded:
                roi = self._crop_with_margin(frame, box)
                if roi is not None and roi.size > 0:
                    try:
                        prediction = self.model.predict(roi)
                    except Exception:
                        prediction = None
            results.append(FaceResult(box=box, prediction=prediction))

        # Orden estable de izquierda a derecha para que el overlay no parpadee.
        results.sort(key=lambda item: (item.box[1] // 40, item.box[0]))
        return results

    # --------------------------------------------------------------- dibujo

    def _label_for(self, face: FaceResult) -> Tuple[str, float]:
        prediction = face.prediction
        if prediction is None:
            return NO_MODEL_LABEL, 0.0
        gender = prediction.gender
        age = prediction.age
        if prediction.gender_conf < self.conf_threshold:
            gender = "Género?"
        if prediction.age_conf < self.conf_threshold:
            age = "Edad?"
        percent = int(round(max(0.0, min(1.0, prediction.confidence)) * 100))
        return f"{gender} | {age} ({percent}%)", prediction.confidence

    @staticmethod
    def _draw_text(
        frame: np.ndarray,
        text: str,
        origin: Tuple[int, int],
        scale: float,
        thickness: int,
        color: Tuple[int, int, int],
        text_color: Tuple[int, int, int] = TEXT_COLOR,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = origin
        y = max(y, text_h + baseline + 4)
        x = max(x, 0)
        cv2.rectangle(
            frame,
            (x, y - text_h - baseline - 4),
            (x + text_w + 8, y + baseline),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            text,
            (x + 4, y - baseline // 2 - 1),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    def annotate(
        self,
        frame: np.ndarray,
        faces: Sequence[FaceResult],
        fps: Optional[float] = None,
        camera_ok: bool = True,
    ) -> np.ndarray:
        """Pinta cajas, etiquetas y un HUD sobre ``frame`` (in-place)."""
        width = frame.shape[1]
        scale = max(0.42, min(0.72, width / 1600.0 + 0.34))
        thickness = max(1, int(round(width / 640.0)))

        for face in faces:
            x, y, w, h = face.box
            color = BOX_COLOR if face.prediction is not None else BOX_COLOR_MUTED
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness + 1)
            label, _ = self._label_for(face)
            self._draw_text(
                frame,
                label,
                (x, max(0, y - 6)),
                scale,
                thickness,
                color,
                TEXT_COLOR,
            )

        hud_parts = [f"Rostros: {len(faces)}"]
        if fps is not None:
            hud_parts.append(f"FPS: {fps:.1f}")
        if not camera_ok:
            hud_parts.append("CAMARA OFFLINE")
        self._draw_text(
            frame,
            "  |  ".join(hud_parts),
            (12, 26),
            max(0.5, scale),
            thickness,
            (28, 32, 40),
            (235, 240, 245),
        )
        return frame
