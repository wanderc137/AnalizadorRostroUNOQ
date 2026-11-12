"""Captura de vídeo desde la cámara USB (Logitech Brio) en un hilo dedicado.

Mantiene siempre el último frame disponible y se reconecta si el dispositivo
desaparece. Los lectores obtienen una copia segura del frame.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera:
    """Wrapper thread-safe de ``cv2.VideoCapture``."""

    def __init__(
        self,
        source: int | str = 0,
        width: int = 640,
        height: int = 480,
        fps_target: int = 15,
        reconnect_delay: float = 2.0,
    ) -> None:
        self.source = source
        self._is_url = isinstance(source, str) and "://" in source
        self.index = 0 if self._is_url else int(source)
        self.width = int(width)
        self.height = int(height)
        self.fps_target = int(fps_target)
        self.reconnect_delay = float(reconnect_delay)

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray] = None
        self._frame_ts: float = 0.0
        self._fps: float = 0.0
        self._prev_ts: Optional[float] = None
        self._failures: int = 0
        self._last_error: str = "Cámara no inicializada"
        self._frame_lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._running = False

    # ------------------------------------------------------------------ API

    def start(self) -> bool:
        """Arranca el hilo de captura (idempotente).

        Cada arranque usa un ``threading.Event`` propio: si un hilo anterior
        aún está terminando, no se "resucita" al limpiar el evento de parada.
        """
        if self._running and self._thread is not None and self._thread.is_alive():
            return True
        stop_event = threading.Event()
        self._stop = stop_event
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            args=(stop_event,),
            name="camera-capture",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self, join_timeout: float = 3.0) -> None:
        """Detiene el hilo, hace join acotado y libera ``VideoCapture``."""
        self._running = False
        self._stop.set()
        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
        # Se libera siempre, incluso si el hilo no terminó a tiempo, para que
        # el dispositivo quede disponible para un start posterior.
        self._release()
        with self._frame_lock:
            self._frame = None
            self._frame_ts = 0.0
            self._fps = 0.0
        self._prev_ts = None
        self._last_error = "Cámara detenida"

    @property
    def running(self) -> bool:
        return self._running

    @property
    def is_ok(self) -> bool:
        """True si hay frames frescos del dispositivo."""
        if not self._running:
            return False
        with self._frame_lock:
            ts = self._frame_ts
        return ts > 0.0 and (time.monotonic() - ts) < 3.0

    @property
    def fps(self) -> float:
        return round(self._fps, 1)

    @property
    def resolution(self) -> Tuple[int, int]:
        with self._frame_lock:
            frame = self._frame
        if frame is not None:
            return int(frame.shape[1]), int(frame.shape[0])
        return self.width, self.height

    @property
    def last_error(self) -> str:
        return self._last_error

    def read(self) -> Optional[np.ndarray]:
        """Devuelve una copia del último frame o ``None``."""
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def _fit_frame(self, frame: np.ndarray) -> np.ndarray:
        """Reduce el frame al tamaño máximo configurado conservando aspecto.

        Las fuentes IP (DroidCam, IP Webcam, etc.) suelen ignorar
        ``CAP_PROP_FRAME_WIDTH``/``HEIGHT`` y entregan 1920x1080. Procesar y
        recomprimir ese tamaño en la UNO Q es muy costoso; aquí se limita a
        ``width`` x ``height`` antes de publicarlo.
        """
        height, width = frame.shape[:2]
        max_width, max_height = int(self.width), int(self.height)
        if width <= max_width and height <= max_height:
            return frame
        scale = min(max_width / width, max_height / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # -------------------------------------------------------------- interno

    def _open(self) -> bool:
        with self._cap_lock:
            self._release_locked()
            if self._is_url:
                backends = [cv2.CAP_FFMPEG, cv2.CAP_ANY]
            else:
                backends = [cv2.CAP_ANY]
                if sys.platform.startswith("linux") and hasattr(cv2, "CAP_V4L2"):
                    backends.insert(0, cv2.CAP_V4L2)
            for backend in backends:
                try:
                    cap = cv2.VideoCapture(self.source, backend)
                except Exception:  # pragma: no cover - backend no disponible
                    continue
                if not cap.isOpened():
                    cap.release()
                    continue
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps_target)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._cap = cap
                self._failures = 0
                self._last_error = ""
                return True
            if self._is_url:
                self._last_error = (
                    "No se pudo abrir la fuente de cámara configurada en CAMERA_SOURCE. "
                    "Verifica la URL, que el teléfono/cámara esté en la misma red y el puerto."
                )
            else:
                self._last_error = (
                    f"No se pudo abrir /dev/video{self.index}. "
                    "Verifica que la cámara USB esté conectada y CAMERA_INDEX."
                )
            return False

    def _release_locked(self) -> None:
        cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:  # pragma: no cover
                pass

    def _release(self) -> None:
        with self._cap_lock:
            self._release_locked()

    def _loop(self, stop_event: threading.Event) -> None:
        while not stop_event.is_set():
            with self._cap_lock:
                cap = self._cap
            if cap is None or not cap.isOpened():
                if not self._open():
                    stop_event.wait(self.reconnect_delay)
                    continue
                with self._cap_lock:
                    cap = self._cap
                if cap is None:
                    continue

            try:
                ok, frame = cap.read()
            except Exception as exc:  # pragma: no cover - error de driver
                ok, frame = False, None
                self._last_error = f"Error de captura: {exc}"

            if not ok or frame is None:
                self._failures += 1
                if self._failures >= 20:
                    self._last_error = "Lectura de cámara fallida; reconectando..."
                    self._release()
                    stop_event.wait(self.reconnect_delay)
                else:
                    stop_event.wait(0.02)
                continue

            frame = self._fit_frame(frame)

            now = time.monotonic()
            with self._frame_lock:
                self._frame = frame
                self._frame_ts = now
                if self._prev_ts is not None:
                    delta = now - self._prev_ts
                    if delta > 0:
                        instant = 1.0 / delta
                        self._fps = (
                            instant if self._fps <= 0.0 else 0.9 * self._fps + 0.1 * instant
                        )
            self._prev_ts = now
            self._failures = 0
            self._last_error = ""

        # La liberación la realiza siempre ``stop()``. Así un hilo antiguo que
        # termina tarde no puede liberar el VideoCapture de un start posterior.

    def status(self) -> dict:
        width, height = self.resolution
        return {
            "camera_ok": self.is_ok,
            "running": self._running,
            "camera_index": self.index,
            "camera_type": "url" if self._is_url else "usb",
            "fps": self.fps,
            "resolution": [width, height],
            "error": self._last_error,
        }
