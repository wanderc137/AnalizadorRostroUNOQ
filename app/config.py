"""Configuración central de AnalizadorRostroUNOQ.

Lee variables de entorno (y `.env`) con valores por defecto pensados para
la Arduino UNO Q (MPU Qualcomm QRB2210, Linux ARM64).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

LOG = logging.getLogger("analizador.config")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Orden fijo de las 8 clases de edad del modelo propio (UTKFace).
DEFAULT_AGE_BUCKETS = "0-2,3-9,10-19,20-29,30-39,40-49,50-59,60+"

VALID_BACKENDS = ("auto", "utkface_onnx", "caffe")

try:  # python-dotenv es opcional: hay un parser mínimo de respaldo.
    from dotenv import load_dotenv as _dotenv_load
except Exception:  # pragma: no cover - entorno sin python-dotenv
    _dotenv_load = None


def _manual_load_env(path: Path) -> None:
    """Parser mínimo de `.env` (KEY=VALUE) usado si falta python-dotenv."""
    try:
        raw = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in raw:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_env(env_file: Optional[Path] = None) -> None:
    candidates = [Path(env_file)] if env_file else [ENV_FILE]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if _dotenv_load is not None:
            _dotenv_load(candidate, override=False)
        else:
            _manual_load_env(candidate)
        return


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _env_int(name: str, default: int, warnings: List[str]) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(float(raw.strip()))
    except (TypeError, ValueError):
        warnings.append(f"{name}={raw!r} no es un entero; se usa {default}.")
        return default


def _env_float(name: str, default: float, warnings: List[str]) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except (TypeError, ValueError):
        warnings.append(f"{name}={raw!r} no es un número; se usa {default}.")
        return default


def _resolve_path(raw: str) -> Optional[Path]:
    """Resuelve rutas relativas contra la raíz del proyecto y normaliza ``..``."""
    raw = raw.strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return path.resolve()
    except OSError:  # pragma: no cover - sistemas de archivos exóticos
        return path.absolute()


def _env_path(name: str, default: str) -> Optional[Path]:
    raw = os.environ.get(name)
    if raw is None:
        raw = default
    return _resolve_path(raw)


def _is_within(root: Path, candidate: Path) -> bool:
    """True si ``candidate`` está dentro de ``root`` (sin resolvedores mágicos)."""
    try:
        candidate.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_snap_folder(raw: str, warnings: List[str]) -> Tuple[Optional[Path], str]:
    """Resuelve ``SNAP_FOLDER`` y rechaza rutas que escapen del proyecto."""
    path = _resolve_path(raw)
    if path is None:
        return None, ""
    root = PROJECT_ROOT.resolve()
    if not _is_within(root, path):
        message = (
            f"SNAP_FOLDER={raw!r} resuelve a {path}, fuera de {root}; "
            "se ignora por seguridad."
        )
        warnings.append(message)
        LOG.error(message)
        return None, message
    return path, ""


@dataclass
class Settings:
    """Configuración efectiva de la aplicación."""

    camera_index: int = 0
    width: int = 640
    height: int = 480
    fps_target: int = 15
    host: str = "0.0.0.0"
    port: int = 8000
    model_backend: str = "auto"
    utkface_onnx_path: Optional[Path] = None
    caffe_age_proto: Optional[Path] = None
    caffe_age_model: Optional[Path] = None
    caffe_gender_proto: Optional[Path] = None
    caffe_gender_model: Optional[Path] = None
    infer_every_n: int = 5
    conf_threshold: float = 0.6
    haar_path: Optional[Path] = None
    snap_folder: Optional[Path] = None
    snap_folder_error: str = ""
    age_buckets: List[str] = field(default_factory=lambda: DEFAULT_AGE_BUCKETS.split(","))
    jpeg_quality: int = 80
    api_token: Optional[str] = None
    snapshot_min_interval: float = 2.0
    warnings: List[str] = field(default_factory=list)

    @property
    def snap_enabled(self) -> bool:
        return self.snap_folder is not None

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_token)

    def ensure_snap_folder(self) -> Optional[Path]:
        """Crea la carpeta de snapshots si está configurada y es segura.

        Se revalida en cada llamada: si la ruta se manipula más tarde o escapa
        del árbol del proyecto, se deshabilita el snapshot y se registra el
        motivo en ``snap_folder_error``.
        """
        if self.snap_folder is None:
            return None
        root = PROJECT_ROOT.resolve()
        try:
            resolved = self.snap_folder.resolve()
        except OSError as exc:  # pragma: no cover - depende del FS de la placa
            self.snap_folder_error = f"No se pudo resolver SNAP_FOLDER: {exc}"
            self.warnings.append(self.snap_folder_error)
            LOG.error(self.snap_folder_error)
            self.snap_folder = None
            return None
        if not _is_within(root, resolved):
            self.snap_folder_error = (
                f"SNAP_FOLDER={str(self.snap_folder)!r} resuelve a {resolved}, "
                f"fuera de {root}; se rechaza por seguridad."
            )
            self.warnings.append(self.snap_folder_error)
            LOG.error(self.snap_folder_error)
            self.snap_folder = None
            return None
        self.snap_folder = resolved
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # pragma: no cover - depende del FS de la placa
            self.warnings.append(f"No se pudo crear {resolved}: {exc}")
            LOG.error("No se pudo crear la carpeta de snapshots %s: %s", resolved, exc)
            return None
        return resolved

    def public_config(self) -> dict:
        """Configuración sin secretos, expuesta por /api/config."""
        return {
            "width": self.width,
            "height": self.height,
            "fps_target": self.fps_target,
            "model_backend": self.model_backend,
            "infer_every_n": self.infer_every_n,
            "conf_threshold": self.conf_threshold,
            "snap_enabled": self.snap_enabled,
            "snapshot_min_interval": self.snapshot_min_interval,
            "auth_enabled": self.auth_enabled,
            "age_buckets": list(self.age_buckets),
            "camera_control": True,
            "snapshot_endpoint": "/api/snapshot",
            "jpeg_quality": self.jpeg_quality,
        }


def load_settings(env_file: Optional[str] = None) -> Settings:
    """Carga la configuración desde `.env` + entorno."""
    _load_env(Path(env_file) if env_file else None)
    warnings: List[str] = []

    backend = _env_str("MODEL_BACKEND", "auto").lower()
    if backend not in VALID_BACKENDS:
        warnings.append(
            f"MODEL_BACKEND={backend!r} no es válido; se usa 'auto'."
        )
        backend = "auto"

    infer_every_n = max(1, _env_int("INFER_EVERY_N", 5, warnings))
    conf_threshold = _env_float("CONF_THRESHOLD", 0.6, warnings)
    conf_threshold = min(max(conf_threshold, 0.0), 1.0)

    buckets_raw = _env_str("AGE_BUCKETS", DEFAULT_AGE_BUCKETS)
    buckets = [item.strip() for item in buckets_raw.split(",") if item.strip()]
    if len(buckets) != 8:
        warnings.append(
            "AGE_BUCKETS debe tener exactamente 8 valores; se usan los del plan."
        )
        buckets = DEFAULT_AGE_BUCKETS.split(",")

    api_token = os.environ.get("API_TOKEN", "").strip() or None
    snapshot_min_interval = max(
        0.0, _env_float("SNAPSHOT_MIN_INTERVAL", 2.0, warnings)
    )
    snap_folder, snap_folder_error = _resolve_snap_folder(
        os.environ.get("SNAP_FOLDER", ""), warnings
    )

    settings = Settings(
        camera_index=max(0, _env_int("CAMERA_INDEX", 0, warnings)),
        width=max(64, _env_int("WIDTH", 640, warnings)),
        height=max(64, _env_int("HEIGHT", 480, warnings)),
        fps_target=max(1, _env_int("FPS_TARGET", 15, warnings)),
        host=_env_str("HOST", "0.0.0.0"),
        port=max(1, _env_int("PORT", 8000, warnings)),
        model_backend=backend,
        utkface_onnx_path=_env_path(
            "UTKFACE_ONNX_PATH", "models/utkface_mobilenetv2.onnx"
        ),
        caffe_age_proto=_env_path(
            "CAFFE_AGE_PROTO", "models/fallback/deploy_age.prototxt"
        ),
        caffe_age_model=_env_path(
            "CAFFE_AGE_MODEL", "models/fallback/age_net.caffemodel"
        ),
        caffe_gender_proto=_env_path(
            "CAFFE_GENDER_PROTO", "models/fallback/deploy_gender.prototxt"
        ),
        caffe_gender_model=_env_path(
            "CAFFE_GENDER_MODEL", "models/fallback/gender_net.caffemodel"
        ),
        infer_every_n=infer_every_n,
        conf_threshold=conf_threshold,
        haar_path=_env_path("HAAR_PATH", ""),
        snap_folder=snap_folder,
        snap_folder_error=snap_folder_error,
        age_buckets=buckets,
        jpeg_quality=min(max(_env_int("JPEG_QUALITY", 80, warnings), 30), 95),
        api_token=api_token,
        snapshot_min_interval=snapshot_min_interval,
        warnings=warnings,
    )
    if settings.auth_enabled:
        LOG.info("API_TOKEN configurado: endpoints de escritura protegidos.")
    settings.ensure_snap_folder()
    for warning in settings.warnings:
        LOG.warning(warning)
    return settings
