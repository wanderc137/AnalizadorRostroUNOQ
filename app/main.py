"""AnalizadorRostroUNOQ — aplicación FastAPI para el MPU Linux de la UNO Q.

Arquitectura de streaming:
  un único worker en background captura frame -> (cada INFER_EVERY_N)
  detecta/clasifica -> dibuja overlay -> codifica JPEG -> publica el último
  JPEG bajo lock. ``/video`` sólo hace yield de ese JPEG (sin inferencia por
  cliente), de modo que N navegadores no multiplican el coste de CPU.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
import math
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Permite `python app/main.py` (standalone) y `python -m app.main`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from app.camera import Camera  # noqa: E402
from app.config import Settings, load_settings  # noqa: E402
from app.detector import FaceDetector, FaceResult  # noqa: E402
from app.models_loader import ModelHandle, load_model  # noqa: E402

LOG = logging.getLogger("analizador")
STATIC_DIR = _PROJECT_ROOT / "static"
MJPEG_BOUNDARY = "frame"


class VideoPipeline:
    """Worker único que produce el JPEG más reciente con overlay."""

    def __init__(
        self,
        settings: Settings,
        camera: Camera,
        detector: Optional[FaceDetector],
        model: ModelHandle,
    ) -> None:
        self.settings = settings
        self.camera = camera
        self.detector = detector
        self.model = model

        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._jpeg: Optional[bytes] = None
        self._frame: Optional[np.ndarray] = None
        self._faces: List[FaceResult] = []
        self._fps: float = 0.0
        self._frame_index: int = 0
        self._last_error: str = ""
        self._placeholder_ts: float = 0.0
        self._started = False
        self._stream_clients = 0

    # ------------------------------------------------------------- ciclo

    def start(self) -> None:
        """Arranca el worker único (idempotente)."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            # Eventos nuevos por ejecución: un stop/start rápido nunca revive
            # al worker anterior ni comparte su señal de parada.
            self._stop = threading.Event()
            self._wake = threading.Event()
            self._started = True
            self._thread = threading.Thread(
                target=self._loop,
                args=(self._stop, self._wake),
                name="video-pipeline",
                daemon=True,
            )
            self._thread.start()

    def stop(self, join_timeout: float = 3.0) -> None:
        """Detiene el worker y espera su salida de forma acotada."""
        with self._lifecycle_lock:
            self._started = False
            self._stop.set()
            self._wake.set()
            thread = self._thread
            self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # --------------------------------------------------- clientes MJPEG

    def add_client(self) -> int:
        """Registra un cliente de ``/video`` y despierta al worker si dormía."""
        with self._lock:
            self._stream_clients += 1
            count = self._stream_clients
        self._wake.set()
        return count

    def remove_client(self) -> int:
        with self._lock:
            self._stream_clients = max(0, self._stream_clients - 1)
            return self._stream_clients

    @property
    def stream_clients(self) -> int:
        with self._lock:
            return self._stream_clients

    # ------------------------------------------------------------- estado

    def set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            frame = self._frame
        return None if frame is None else frame.copy()

    @property
    def faces_count(self) -> int:
        with self._lock:
            return len(self._faces)

    @property
    def fps(self) -> float:
        with self._lock:
            return round(self._fps, 1)

    @property
    def last_error(self) -> str:
        with self._lock:
            return self._last_error

    def status(self) -> dict:
        camera_ok = self.camera.is_ok
        model_loaded = self.model.loaded
        width, height = self.camera.resolution
        with self._lock:
            frame = self._frame
            faces_count = len(self._faces)
            fps = self._fps
            error = self._last_error
            clients = self._stream_clients
        if frame is not None:
            width, height = int(frame.shape[1]), int(frame.shape[0])

        if self.detector is None:
            message = f"Detector no disponible: {error or 'Haar cascade no cargado'}"
        elif not self._started:
            message = "Pipeline detenido"
        elif not self.is_running:
            message = "Pipeline detenido inesperadamente"
        elif not self.camera.running:
            message = "Cámara detenida"
        elif not camera_ok:
            message = error or self.camera.last_error or "Cámara no disponible"
        elif not model_loaded:
            message = "Modelo no cargado: modo solo detección"
        else:
            message = "OK"

        return {
            "camera_ok": camera_ok,
            "model_backend": self.model.name,
            "model_backend_requested": self.model.requested,
            "model_loaded": model_loaded,
            "degraded": (not camera_ok) or (not model_loaded),
            "fps": round(float(fps), 1),
            "faces_count": int(faces_count),
            "resolution": [int(width), int(height)],
            "message": message,
            "model_error": self.model.error if not model_loaded else "",
            "camera_error": "" if camera_ok else self.camera.last_error,
            "pipeline_error": error,
            "pipeline_running": self.is_running,
            "stream_clients": int(clients),
        }

    # ------------------------------------------------------------ worker

    def _build_placeholder(self) -> np.ndarray:
        width, height = self.settings.width, self.settings.height
        frame = np.full((height, width, 3), 16, dtype=np.uint8)
        cv2.line(
            frame, (0, 0), (width, height), (26, 32, 44), 2
        )
        cv2.line(
            frame, (width, 0), (0, height), (26, 32, 44), 2
        )
        scale = max(0.5, min(0.9, width / 1100.0 + 0.4))
        cv2.putText(
            frame,
            "CAMARA NO DISPONIBLE",
            (int(width * 0.08), int(height * 0.46)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale + 0.15,
            (90, 190, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "Conecta la Logitech Brio o revisa CAMERA_INDEX",
            (int(width * 0.08), int(height * 0.56)),
            cv2.FONT_HERSHEY_SIMPLEX,
            max(0.4, scale - 0.15),
            (170, 180, 195),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _publish(self, frame: np.ndarray, jpeg: bytes, faces: List[FaceResult]) -> None:
        with self._lock:
            self._frame = frame
            self._jpeg = jpeg
            self._faces = faces

    def _encode(self, frame: np.ndarray) -> Optional[bytes]:
        quality = int(self.settings.jpeg_quality)
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        )
        if not ok:
            return None
        return buffer.tobytes()

    def _idle_step(self) -> None:
        """Sin clientes MJPEG: conserva el frame para snapshots, sin codificar.

        No se ejecuta Haar ni la red, y tampoco se codifica JPEG: el coste en
        reposo se reduce a una lectura de cámara cada 250 ms. En cuanto hay un
        cliente, ``add_client()`` despierta al worker y el stream se reanuda.
        """
        frame = self.camera.read()
        with self._lock:
            if frame is not None:
                self._frame = frame
            self._faces = []
            self._fps = 0.0
            self._jpeg = None

    def _loop(self, stop: threading.Event, wake: threading.Event) -> None:
        period = 1.0 / max(1, self.settings.fps_target)
        every_n = max(1, self.settings.infer_every_n)
        prev_ts = time.monotonic()

        while not stop.is_set():
            try:
                iteration_start = time.monotonic()

                # ------------------------------------------------ sin clientes
                if self.stream_clients <= 0:
                    self._idle_step()
                    wake.wait(0.25)
                    wake.clear()
                    prev_ts = time.monotonic()
                    continue

                # ------------------------------------------------ con clientes
                frame = self.camera.read()
                if frame is None:
                    now = time.monotonic()
                    if self._jpeg is None or (now - self._placeholder_ts) > 1.0:
                        placeholder = self._build_placeholder()
                        jpeg = self._encode(placeholder)
                        if jpeg is not None:
                            self._publish(placeholder, jpeg, [])
                        self._placeholder_ts = now
                    with self._lock:
                        self._fps = 0.0
                    wake.wait(0.25)
                    wake.clear()
                    continue

                if self._last_error.startswith("Error en pipeline"):
                    with self._lock:
                        self._last_error = ""

                self._frame_index += 1
                faces: List[FaceResult]
                if self.detector is not None and (
                    self._frame_index % every_n == 0 or not self._faces
                ):
                    try:
                        faces = self.detector.detect(frame)
                    except Exception as exc:  # pragma: no cover - defensivo
                        faces = []
                        with self._lock:
                            self._last_error = f"Error en detección: {exc}"
                    with self._lock:
                        self._faces = faces
                with self._lock:
                    faces = list(self._faces)

                annotated = frame.copy()
                if self.detector is not None:
                    self.detector.annotate(
                        annotated,
                        faces,
                        fps=self.fps,
                        camera_ok=True,
                    )

                jpeg = self._encode(annotated)
                if jpeg is not None:
                    self._publish(annotated, jpeg, faces)

                now = time.monotonic()
                delta = now - prev_ts
                prev_ts = now
                if delta > 0:
                    instant = 1.0 / delta
                    with self._lock:
                        self._fps = (
                            instant
                            if self._fps <= 0.0
                            else 0.85 * self._fps + 0.15 * instant
                        )

                elapsed = time.monotonic() - iteration_start
                wait = period - elapsed
                if wait > 0:
                    stop.wait(wait)
            except Exception as exc:  # I-003: el worker nunca debe morir
                LOG.exception("Error inesperado en el worker de vídeo")
                with self._lock:
                    self._last_error = f"Error en pipeline: {exc}"
                wake.wait(0.5)
                wake.clear()


def _build_pipeline(settings: Settings) -> tuple:
    camera = Camera(
        index=settings.camera_index,
        width=settings.width,
        height=settings.height,
        fps_target=settings.fps_target,
    )
    model = load_model(settings)
    detector: Optional[FaceDetector] = None
    detector_error = ""
    try:
        detector = FaceDetector(
            model,
            haar_path=settings.haar_path,
            conf_threshold=settings.conf_threshold,
        )
    except Exception as exc:
        detector_error = str(exc)
        LOG.error("No se pudo inicializar el detector Haar: %s", exc)
    pipeline = VideoPipeline(settings, camera, detector, model)
    if detector_error:
        pipeline.set_error(detector_error)
    return camera, model, detector, pipeline


def _check_api_token(cfg: Settings, request: Request) -> None:
    """Exige ``X-Api-Token`` en endpoints de escritura si ``API_TOKEN`` existe."""
    expected = cfg.api_token
    if not expected:
        return
    provided = request.headers.get("x-api-token", "")
    if not provided or not hmac.compare_digest(
        provided.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(
            status_code=401,
            detail=(
                "Token de API ausente o inválido. "
                "Envía el header X-Api-Token con el valor de API_TOKEN."
            ),
            headers={"WWW-Authenticate": 'ApiToken realm="AnalizadorRostroUNOQ"'},
        )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    cfg = settings or load_settings()
    camera, model, detector, pipeline = _build_pipeline(cfg)

    control_lock = asyncio.Lock()
    snapshot_lock = threading.Lock()
    snapshot_state = {"last_ts": 0.0}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        async with control_lock:
            camera.start()
            pipeline.start()
        LOG.info(
            "Analizador listo en http://%s:%s (backend=%s, camara=/dev/video%s, auth=%s)",
            cfg.host,
            cfg.port,
            model.name,
            cfg.camera_index,
            "on" if cfg.auth_enabled else "off",
        )
        try:
            yield
        finally:
            async with control_lock:
                await asyncio.to_thread(pipeline.stop)
                await asyncio.to_thread(camera.stop)

    app = FastAPI(
        title="AnalizadorRostroUNOQ",
        description=(
            "Detección de rostros y estimación de género/rango de edad "
            "en la Arduino UNO Q (MPU Linux)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = cfg
    app.state.camera = camera
    app.state.model = model
    app.state.detector = detector
    app.state.pipeline = pipeline

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        index_file = STATIC_DIR / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=500, detail="static/index.html no existe")
        return FileResponse(index_file)

    @app.get("/api/status")
    async def api_status():
        return JSONResponse(pipeline.status())

    @app.get("/api/config")
    async def api_config():
        return JSONResponse(cfg.public_config())

    @app.get("/healthz", include_in_schema=False)
    async def healthz():
        return {"ok": True}

    @app.post("/api/camera/start")
    async def api_camera_start(request: Request):
        _check_api_token(cfg, request)
        async with control_lock:
            if camera.running and pipeline.is_running and camera.is_ok:
                return JSONResponse(
                    {
                        "ok": True,
                        "message": "La cámara ya está activa",
                        "camera": camera.status(),
                        "pipeline_running": True,
                    }
                )

            # Arranque coherente: se detiene pipeline -> cámara y se reconstruye
            # el estado desde cero para evitar workers huérfanos o doble start.
            await asyncio.to_thread(pipeline.stop)
            await asyncio.to_thread(camera.stop)
            await asyncio.to_thread(camera.start)

            deadline = time.monotonic() + 1.5
            while time.monotonic() < deadline:
                if camera.is_ok:
                    break
                await asyncio.sleep(0.05)

            if not camera.is_ok:
                detail = (
                    "No se pudo abrir la cámara en 1.5 s. "
                    + (
                        camera.last_error
                        or "Verifica CAMERA_INDEX y la conexión USB de la Brio."
                    )
                )
                await asyncio.to_thread(camera.stop)
                LOG.error("Start de cámara fallido: %s", detail)
                raise HTTPException(status_code=503, detail=detail)

            pipeline.start()
            LOG.info("Cámara y pipeline iniciados desde /api/camera/start")
            return JSONResponse(
                {
                    "ok": True,
                    "camera": camera.status(),
                    "pipeline_running": pipeline.is_running,
                }
            )

    @app.post("/api/camera/stop")
    async def api_camera_stop(request: Request):
        _check_api_token(cfg, request)
        async with control_lock:
            # Primero el worker (deja de leer/emitir), luego la captura.
            await asyncio.to_thread(pipeline.stop)
            await asyncio.to_thread(camera.stop)
            LOG.info("Cámara y pipeline detenidos desde /api/camera/stop")
            return JSONResponse(
                {
                    "ok": True,
                    "camera": camera.status(),
                    "pipeline_running": pipeline.is_running,
                }
            )

    @app.post("/api/snapshot")
    async def api_snapshot(request: Request):
        _check_api_token(cfg, request)

        # Rate-limit atómico antes de cualquier trabajo de disco.
        now = time.monotonic()
        interval = max(0.0, cfg.snapshot_min_interval)
        with snapshot_lock:
            elapsed = now - snapshot_state["last_ts"]
            if snapshot_state["last_ts"] > 0.0 and elapsed < interval:
                retry_after = max(1, int(math.ceil(interval - elapsed)))
                raise HTTPException(
                    status_code=429,
                    detail=f"Espera {retry_after} s antes de guardar otro snapshot.",
                    headers={"Retry-After": str(retry_after)},
                )
            snapshot_state["last_ts"] = now

        folder = cfg.ensure_snap_folder()
        if folder is None:
            if cfg.snap_folder_error:
                LOG.error("Snapshot rechazado: %s", cfg.snap_folder_error)
                raise HTTPException(
                    status_code=400,
                    detail=f"SNAP_FOLDER inválido: {cfg.snap_folder_error}",
                )
            raise HTTPException(
                status_code=503,
                detail="Snapshot deshabilitado. Configura SNAP_FOLDER en .env",
            )
        frame = pipeline.latest_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="No hay frame disponible")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        target = folder / f"snapshot_{stamp}.jpg"
        try:
            saved = False
            try:
                from PIL import Image  # type: ignore

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                Image.fromarray(rgb).save(str(target), "JPEG", quality=92)
                saved = True
            except Exception:
                saved = False
            if not saved:
                jpeg = pipeline.latest_jpeg()
                if jpeg is None:
                    ok, buffer = cv2.imencode(".jpg", frame)
                    if not ok:
                        raise RuntimeError("No se pudo codificar el snapshot")
                    jpeg = buffer.tobytes()
                target.write_bytes(jpeg)
        except Exception as exc:  # pragma: no cover - depende del FS
            raise HTTPException(
                status_code=500, detail=f"No se pudo guardar el snapshot: {exc}"
            ) from exc
        return JSONResponse({"ok": True, "path": str(target)})

    @app.get("/video")
    async def video_feed():
        async def stream():
            pipeline.add_client()
            last: Optional[bytes] = None
            try:
                while True:
                    jpeg = pipeline.latest_jpeg()
                    if jpeg is not None and jpeg is not last:
                        last = jpeg
                        header = (
                            f"--{MJPEG_BOUNDARY}\r\n"
                            "Content-Type: image/jpeg\r\n"
                            f"Content-Length: {len(jpeg)}\r\n\r\n"
                        ).encode("ascii")
                        yield header + jpeg + b"\r\n"
                    elif jpeg is None:
                        await asyncio.sleep(0.1)
                    else:
                        await asyncio.sleep(0.03)
            finally:
                pipeline.remove_client()

        return StreamingResponse(
            stream(),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Connection": "close",
            },
        )

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _settings = load_settings()
    uvicorn.run(
        app,
        host=_settings.host,
        port=_settings.port,
        log_level="info",
    )
