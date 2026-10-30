"""Carga de modelos de género/edad y predicción sobre un rostro (ROI BGR).

Dos backends soportados:

* ``utkface_onnx``: MobileNetV2 propia (UTKFace) exportada a ONNX 224x224.
  Entrada normalizada (x / 127.5 - 1), salidas ``gender`` (sigmoid) y
  ``age`` (softmax de 8 clases). ``gender``: 0 = Masculino, 1 = Femenino.
* ``caffe``: modelos preentrenados Levi & Hassner 227x227 (age_net /
  gender_net). Usan SUS PROPIOS buckets de edad clásicos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOG = logging.getLogger("analizador.models")

ONNX_INPUT_SIZE: Tuple[int, int] = (224, 224)
ONNX_SCALE = 1.0 / 127.5
ONNX_MEAN = (127.5, 127.5, 127.5)

CAFFE_INPUT_SIZE: Tuple[int, int] = (227, 227)
CAFFE_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
CAFFE_GENDER_LABELS: Tuple[str, str] = ("Masculino", "Femenino")
# Buckets propios del modelo Caffe (Levi & Hassner). NO son los de UTKFace.
CAFFE_AGE_BUCKETS: Tuple[str, ...] = (
    "0-2",
    "4-6",
    "8-12",
    "15-20",
    "25-32",
    "38-43",
    "48-53",
    "60-100",
)


@dataclass
class FacePrediction:
    """Predicción combinada para un único rostro."""

    gender: str
    age: str
    gender_conf: float
    age_conf: float

    @property
    def confidence(self) -> float:
        return round((float(self.gender_conf) + float(self.age_conf)) / 2.0, 3)

    def as_dict(self) -> dict:
        return {
            "gender": self.gender,
            "age": self.age,
            "gender_confidence": round(float(self.gender_conf), 3),
            "age_confidence": round(float(self.age_conf), 3),
            "confidence": self.confidence,
        }


def _sigmoid(value: float) -> float:
    value = float(np.clip(value, -60.0, 60.0))
    return 1.0 / (1.0 + float(np.exp(-value)))


def _softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr - float(np.max(arr))
    exp = np.exp(arr)
    total = float(np.sum(exp))
    if total <= 0.0:
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float32)
    return (exp / total).astype(np.float32)


def _is_probability_vector(values: np.ndarray) -> bool:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return False
    if float(np.min(arr)) < -1e-4 or float(np.max(arr)) > 1.0 + 1e-4:
        return False
    return abs(float(np.sum(arr)) - 1.0) < 0.05


class BaseBackend:
    """Interfaz común de los backends de clasificación."""

    name: str = "none"

    def predict(
        self, face_bgr: np.ndarray
    ) -> Optional[FacePrediction]:  # pragma: no cover
        raise NotImplementedError

    def describe(self) -> dict:  # pragma: no cover
        return {"backend": self.name}


class OnnxBackend(BaseBackend):
    """Modelo propio UTKFace exportado a ONNX."""

    name = "utkface_onnx"

    def __init__(self, model_path: Path, age_buckets: Sequence[str]) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"No existe el modelo ONNX: {model_path}")
        self.model_path = model_path
        self.age_buckets: List[str] = list(age_buckets)
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        self._output_names: List[str] = []
        try:
            self._output_names = list(self._net.getUnconnectedOutLayersNames())
        except Exception:  # pragma: no cover - versiones antiguas de OpenCV
            self._output_names = []

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "path": str(self.model_path),
            "input_size": list(ONNX_INPUT_SIZE),
            "outputs": list(self._output_names),
        }

    def _forward(self) -> Optional[List[Tuple[str, np.ndarray]]]:
        """Ejecuta el forward una sola vez; ante error devuelve ``None``.

        No se reintenta la misma llamada: un fallo de OpenCV DNN no se
        recupera repitiendo la operación sin cambiar entradas ni estado.
        El pipeline trata ``None`` como predicción degradada (sin etiqueta).
        """
        names = self._output_names
        try:
            if len(names) >= 2:
                outputs = self._net.forward(names)
                return list(zip(names, outputs))
            raw = self._net.forward()
        except cv2.error as exc:
            LOG.error("Fallo en forward ONNX (%s); se degrada la predicción.", exc)
            return None
        if isinstance(raw, (tuple, list)):
            return [
                (names[i] if i < len(names) else f"output_{i}", np.asarray(item))
                for i, item in enumerate(raw)
            ]
        return [(names[0] if names else "output", np.asarray(raw))]

    def _split_outputs(
        self, pairs: Sequence[Tuple[str, np.ndarray]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        gender_vec: Optional[np.ndarray] = None
        age_vec: Optional[np.ndarray] = None
        for name, array in pairs:
            flat = np.asarray(array, dtype=np.float32).reshape(-1)
            lname = name.lower()
            if "gender" in lname and gender_vec is None:
                gender_vec = flat
                continue
            if "age" in lname and age_vec is None:
                age_vec = flat
                continue
        if gender_vec is None or age_vec is None:
            for name, array in pairs:
                flat = np.asarray(array, dtype=np.float32).reshape(-1)
                if flat.size == len(self.age_buckets) and age_vec is None:
                    age_vec = flat
                elif flat.size in (1, 2) and gender_vec is None:
                    gender_vec = flat
        if gender_vec is None or age_vec is None:
            raise RuntimeError(
                "El ONNX no expone salidas compatibles con 'gender' y 'age'."
            )
        if age_vec.size != len(self.age_buckets):
            raise RuntimeError(
                f"Salida de edad con {age_vec.size} clases; se esperaban "
                f"{len(self.age_buckets)} (AGE_BUCKETS)."
            )
        return gender_vec, age_vec

    def predict(self, face_bgr: np.ndarray) -> Optional[FacePrediction]:
        blob = cv2.dnn.blobFromImage(
            face_bgr,
            ONNX_SCALE,
            ONNX_INPUT_SIZE,
            ONNX_MEAN,
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        pairs = self._forward()
        if pairs is None:
            return None
        gender_vec, age_vec = self._split_outputs(pairs)

        if gender_vec.size == 2:
            probs = (
                gender_vec
                if _is_probability_vector(gender_vec)
                else _softmax(gender_vec)
            )
            female_prob = float(probs[1])
        else:
            raw = float(gender_vec.reshape(-1)[0])
            female_prob = raw if 0.0 <= raw <= 1.0 else _sigmoid(raw)
        gender_index = 1 if female_prob >= 0.5 else 0
        gender_label = CAFFE_GENDER_LABELS[gender_index]
        gender_conf = female_prob if gender_index == 1 else 1.0 - female_prob

        age_probs = (
            age_vec if _is_probability_vector(age_vec) else _softmax(age_vec)
        )
        age_index = int(np.argmax(age_probs))
        age_label = (
            self.age_buckets[age_index]
            if age_index < len(self.age_buckets)
            else f"clase {age_index}"
        )
        age_conf = float(age_probs[age_index])

        return FacePrediction(
            gender=gender_label,
            age=age_label,
            gender_conf=round(float(gender_conf), 4),
            age_conf=round(age_conf, 4),
        )


class CaffeBackend(BaseBackend):
    """Modelos Caffe preentrenados (Levi & Hassner)."""

    name = "caffe"

    def __init__(
        self,
        age_proto: Path,
        age_model: Path,
        gender_proto: Path,
        gender_model: Path,
    ) -> None:
        paths = {
            "age_proto": Path(age_proto),
            "age_model": Path(age_model),
            "gender_proto": Path(gender_proto),
            "gender_model": Path(gender_model),
        }
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Faltan archivos Caffe: " + ", ".join(missing)
            )
        self.paths = paths
        self._age_net = cv2.dnn.readNetFromCaffe(
            str(paths["age_proto"]), str(paths["age_model"])
        )
        self._gender_net = cv2.dnn.readNetFromCaffe(
            str(paths["gender_proto"]), str(paths["gender_model"])
        )

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "path": str(self.paths["gender_model"].parent),
            "input_size": list(CAFFE_INPUT_SIZE),
            "age_buckets": list(CAFFE_AGE_BUCKETS),
        }

    def _blob(self, face_bgr: np.ndarray) -> np.ndarray:
        return cv2.dnn.blobFromImage(
            face_bgr,
            1.0,
            CAFFE_INPUT_SIZE,
            CAFFE_MEAN,
            swapRB=False,
            crop=False,
        )

    def predict(self, face_bgr: np.ndarray) -> FacePrediction:
        blob = self._blob(face_bgr)

        self._gender_net.setInput(blob)
        gender_probs = np.asarray(
            self._gender_net.forward(), dtype=np.float32
        ).reshape(-1)
        if gender_probs.size < 2:
            raise RuntimeError("Salida de género Caffe inesperada.")
        gender_index = int(np.argmax(gender_probs[:2]))
        # gender_net clásico: 0 = Male, 1 = Female.
        gender_label = CAFFE_GENDER_LABELS[gender_index]
        gender_conf = float(gender_probs[gender_index])

        self._age_net.setInput(blob)
        age_probs = np.asarray(
            self._age_net.forward(), dtype=np.float32
        ).reshape(-1)
        age_index = int(np.argmax(age_probs))
        age_label = (
            CAFFE_AGE_BUCKETS[age_index]
            if age_index < len(CAFFE_AGE_BUCKETS)
            else f"clase {age_index}"
        )
        age_conf = float(age_probs[age_index])

        return FacePrediction(
            gender=gender_label,
            age=age_label,
            gender_conf=round(gender_conf, 4),
            age_conf=round(age_conf, 4),
        )


@dataclass
class ModelHandle:
    """Resultado de la carga: backend activo o modo degradado."""

    requested: str
    backend: Optional[BaseBackend] = None
    error: str = ""

    @property
    def name(self) -> str:
        return self.backend.name if self.backend is not None else "none"

    @property
    def loaded(self) -> bool:
        return self.backend is not None

    def predict(self, face_bgr: np.ndarray) -> Optional[FacePrediction]:
        if self.backend is None:
            return None
        return self.backend.predict(face_bgr)

    def describe(self) -> dict:
        info = {
            "requested": self.requested,
            "model_backend": self.name,
            "model_loaded": self.loaded,
            "error": self.error,
        }
        if self.backend is not None:
            info.update(self.backend.describe())
        return info


def _try_onnx(settings, errors: List[str]) -> Optional[OnnxBackend]:
    path = settings.utkface_onnx_path
    if path is None:
        errors.append("UTKFACE_ONNX_PATH no configurado.")
        return None
    if not Path(path).is_file():
        errors.append(f"Modelo ONNX no encontrado: {path}")
        return None
    try:
        return OnnxBackend(Path(path), settings.age_buckets)
    except Exception as exc:
        errors.append(f"Error cargando ONNX: {exc}")
        return None


def _try_caffe(settings, errors: List[str]) -> Optional[CaffeBackend]:
    required = [
        settings.caffe_age_proto,
        settings.caffe_age_model,
        settings.caffe_gender_proto,
        settings.caffe_gender_model,
    ]
    if any(path is None for path in required):
        errors.append("Rutas Caffe incompletas en la configuración.")
        return None
    missing = [str(p) for p in required if not Path(p).is_file()]
    if missing:
        errors.append("Faltan modelos Caffe: " + ", ".join(missing))
        return None
    try:
        return CaffeBackend(
            required[0], required[1], required[2], required[3]
        )
    except Exception as exc:
        errors.append(f"Error cargando Caffe: {exc}")
        return None


def load_model(settings) -> ModelHandle:
    """Carga el backend según ``MODEL_BACKEND``.

    ``auto`` intenta ONNX propio y luego Caffe. Si ninguno está disponible
    devuelve un handle sin backend (modo degradado: solo detección).
    """
    requested = getattr(settings, "model_backend", "auto")
    errors: List[str] = []

    if requested == "utkface_onnx":
        backend = _try_onnx(settings, errors)
    elif requested == "caffe":
        backend = _try_caffe(settings, errors)
    else:  # auto
        backend = _try_onnx(settings, errors)
        if backend is None:
            backend = _try_caffe(settings, errors)

    if backend is not None:
        return ModelHandle(requested=requested, backend=backend)

    message = "; ".join(errors) if errors else "No hay modelos configurados."
    return ModelHandle(requested=requested, backend=None, error=message)
