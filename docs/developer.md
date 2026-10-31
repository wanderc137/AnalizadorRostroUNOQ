# AnalizadorRostroUNOQ — Guía del desarrollador

Documento técnico interno: arquitectura, concurrencia, contrato de API, backends, scripts, entrenamiento y vías de extensión.

> **Contexto imprescindible:** esto **no es un sketch Arduino**. Es una aplicación Python/FastAPI que corre en el **MPU Linux** (Qualcomm QRB2210, ARM64, Debian) de la UNO Q. El MCU STM32U585 con Zephyr **no interviene**: no hay `setup()`, `loop()`, GPIO ni `arduino-cli`.

---

## 1. Mapa de módulos

```text
app/
├── __init__.py          vacío / marca de paquete
├── config.py            Settings + load_settings() + validación + seguridad de rutas
├── camera.py            Camera: hilo de captura V4L2, reconexión, lectura thread-safe
├── detector.py          FaceDetector: Haar + crop + overlay; FaceResult
├── models_loader.py     BaseBackend / OnnxBackend / CaffeBackend / ModelHandle
└── main.py              FastAPI, VideoPipeline, endpoints, lifespan

python/main.py           entrypoint App Lab: uvicorn.run("app.main:app", workers=1)
static/                  UI vanilla (sin build)
scripts/                 descarga, setup, deploy, run
training/                entrenamiento UTKFace (fuera de la placa)
models/                  ONNX propio (release v1.0.0, gitignored) + fallback Caffe (gitignored)
```

### Flujo de importación

```text
python/main.py
  └── app.config.load_settings()
  └── uvicorn("app.main:app")
        └── app.main.create_app()
              ├── app.camera.Camera
              ├── app.models_loader.load_model()
              ├── app.detector.FaceDetector
              └── app.main.VideoPipeline
```

`app/main.py` inserta la raíz del proyecto en `sys.path` para permitir tanto `python app/main.py` como `python -m app.main`.

---

## 2. `app/config.py`

### 2.1 Responsabilidad

Leer `.env` + variables de entorno, aplicar defaults, validar rangos y resolver rutas de forma segura.

### 2.2 Carga de `.env`

- Si `python-dotenv` está instalado: `load_dotenv(path, override=False)`.
- Si no: parser mínimo `KEY=VALUE` que ignora comentarios y `export`, y **no sobrescribe** variables ya presentes en `os.environ`.
- El archivo se busca en `PROJECT_ROOT/.env`.

### 2.3 Helpers

| Función | Comportamiento |
|---|---|
| `_env_str(name, default)` | Recorta; cadena vacía → default. |
| `_env_int(name, default, warnings)` | Acepta `"5"` y `"5.0"` (`int(float(...))`); inválido → warning + default. |
| `_env_float(...)` | Igual para floats. |
| `_resolve_path(raw)` | `expanduser()`; relativas contra `PROJECT_ROOT`; `resolve()` para normalizar `..`. |
| `_is_within(root, candidate)` | `candidate.relative_to(root)`; `True` si está dentro. |
| `_resolve_snap_folder(raw, warnings)` | Resuelve y rechaza rutas fuera de `PROJECT_ROOT`. |

### 2.4 `Settings` (dataclass)

Campos principales: `camera_index`, `width`, `height`, `fps_target`, `host`, `port`, `model_backend`, rutas de modelos, `infer_every_n`, `conf_threshold`, `haar_path`, `snap_folder`, `snap_folder_error`, `age_buckets`, `jpeg_quality`, `api_token`, `snapshot_min_interval`, `warnings`.

Propiedades:

| Miembro | Uso |
|---|---|
| `snap_enabled` | `snap_folder is not None`. |
| `auth_enabled` | `bool(api_token)`. |
| `ensure_snap_folder()` | Revalida la ruta en cada llamada, la crea y devuelve `Path` o `None`. Si escapa del proyecto, anula `snap_folder` y llena `snap_folder_error`. |
| `public_config()` | JSON sin secretos para `/api/config`. |

### 2.5 Validaciones

| Variable | Regla |
|---|---|
| `MODEL_BACKEND` | Debe estar en `("auto", "utkface_onnx", "caffe")`; si no, `auto` + warning. |
| `INFER_EVERY_N` | `max(1, valor)`. |
| `CONF_THRESHOLD` | Recortado a `[0, 1]`. |
| `AGE_BUCKETS` | Exactamente 8 elementos; si no, se restauran los defaults. |
| `CAMERA_INDEX` | `max(0, valor)` (el índice real se valida al abrir). |
| `WIDTH` / `HEIGHT` | `max(64, valor)`. |
| `FPS_TARGET` | `max(1, valor)`. |
| `PORT` | `max(1, valor)` (sin techo explícito; Uvicorn fallará si > 65535). |
| `JPEG_QUALITY` | Recortado a `[30, 95]`. |
| `SNAPSHOT_MIN_INTERVAL` | `max(0.0, valor)`. |
| `SNAP_FOLDER` | Debe resolver dentro de `PROJECT_ROOT`. |

### 2.6 Nota de seguridad

`SNAP_FOLDER` se valida **dos veces**: al cargar (`_resolve_snap_folder`) y en cada snapshot (`ensure_snap_folder`). Esto cubre el caso de que la ruta se manipule después del arranque o de que un symlink cambie. La comprobación usa `Path.resolve()` + `relative_to`, por lo que `../` y rutas absolutas externas se rechazan.

---

## 3. `app/camera.py`

### 3.1 Modelo de concurrencia

| Recurso | Protección | Escritor | Lector |
|---|---|---|---|
| `_frame`, `_frame_ts`, `_fps` | `_frame_lock` | hilo `camera-capture` | `read()`, `is_ok`, `resolution`, `status()` |
| `_cap` (`cv2.VideoCapture`) | `_cap_lock` | `_open`, `_release` | `_loop` |
| `_stop` | `threading.Event` nuevo en cada `start()` | `stop()` | `_loop(stop_event)` |

`read()` devuelve **copia** (`frame.copy()`) para que el consumidor no mute el frame compartido.

### 3.2 Ciclo de vida

```text
start()
  └── crea threading.Event propio (no reutiliza el anterior)
  └── lanza hilo daemon "camera-capture"

_loop(stop_event)
  ├── si _cap es None o no está abierto → _open()
  │     ├── CAP_V4L2 (Linux) y luego CAP_ANY
  │     ├── set WIDTH/HEIGHT/FPS/BUFFERSIZE=1
  │     └── si falla → wait(reconnect_delay=2s)
  ├── cap.read()
  │     ├── fallo < 20 → wait(0.02)
  │     └── fallo ≥ 20 → release + wait(2s)  (fuerza reapertura)
  └── éxito → actualiza frame/ts/fps (EMA 0.9/0.1)

stop(timeout=3.0)
  ├── _running=False, _stop.set()
  ├── join(timeout)
  ├── _release()  ← siempre, aunque el hilo no haya terminado
  └── limpia frame/ts/fps y last_error="Cámara detenida"
```

### 3.3 Decisiones relevantes

- **`Event` nuevo por arranque:** si un hilo viejo sigue terminando, no se "resucita" al limpiar el evento compartido.
- **La liberación la hace `stop()`, no el hilo:** así un hilo antiguo que termina tarde no libera el `VideoCapture` de un `start()` posterior.
- **`is_ok`** exige frame con menos de 3 s: detecta cámara congelada, no solo "abierta".
- **Reconexión** tras 20 fallos consecutivos; el umbral evita reabrir por un frame perdido.

### 3.4 API pública

| Miembro | Firma | Propósito |
|---|---|---|
| `start` | `() -> bool` | Arranca el hilo (idempotente). |
| `stop` | `(join_timeout=3.0) -> None` | Detiene, hace join y libera. |
| `read` | `() -> Optional[np.ndarray]` | Copia del último frame BGR. |
| `is_ok` | `@property -> bool` | Hay frame fresco (< 3 s). |
| `running` | `@property -> bool` | El flag de ejecución está activo. |
| `fps` | `@property -> float` | FPS suavizado de captura. |
| `resolution` | `@property -> Tuple[int,int]` | Tamaño real del frame. |
| `status` | `() -> dict` | `camera_ok`, `running`, índice, fps, resolución, error. |

---

## 4. `app/detector.py`

### 4.1 `FaceResult`

```python
@dataclass
class FaceResult:
    box: Tuple[int, int, int, int]      # x, y, w, h
    prediction: Optional[FacePrediction]
    @property
    def confidence(self) -> float       # 0.0 si no hay predicción
```

### 4.2 Carga del cascade

`_resolve_cascade_path(haar_path)`:

- Si `HAAR_PATH` está definido y existe → se usa.
- Si está definido y no existe → `FileNotFoundError`.
- Si está vacío → `cv2.data.haarcascades / "haarcascade_frontalface_default.xml"`.

`FaceDetector.__init__` falla con `RuntimeError` si `CascadeClassifier.empty()`.

### 4.3 `detect(frame)`

```text
frame BGR
  → cvtColor(GRAY)
  → equalizeHist
  → detectMultiScale(scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
  → para cada caja:
        crop con margen 6 % (x0/y0/x1/y1 recortados al frame)
        si model.loaded y ROI válida → model.predict(roi)
  → orden estable: (y // 40, x)   ← evita parpadeo del overlay
```

El orden por bandas de 40 px agrupa rostros en la misma "fila" y luego ordena por X.

### 4.4 Etiquetas

`_label_for(face)`:

```python
if prediction is None:
    return "Rostro detectado | modelo no cargado", 0.0
gender = prediction.gender if prediction.gender_conf >= threshold else "Género?"
age    = prediction.age    if prediction.age_conf    >= threshold else "Edad?"
percent = round(prediction.confidence * 100)   # media de ambas confianzas
return f"{gender} | {age} ({percent}%)", prediction.confidence
```

### 4.5 Overlay

`annotate(frame, faces, fps, camera_ok)` dibuja **in-place**:

- Caja: `BOX_COLOR=(0,255,180)` BGR si hay predicción; `BOX_COLOR_MUTED=(0,190,255)` si no.
- Etiqueta con fondo relleno y texto oscuro; posición `(x, y-6)` con clamp para que no salga del frame.
- HUD superior izquierdo: `Rostros: N | FPS: x.x | CAMARA OFFLINE`.
- Escala y grosor proporcionales al ancho (`width/1600 + 0.34`, clamp 0.42–0.72).

### 4.6 Puntos de extensión

- `scale_factor`, `min_neighbors`, `min_size` son parámetros del constructor, pero `main._build_pipeline` los deja en defaults. Para exponerlos por `.env`, añade campos a `Settings` y pásalos aquí.
- Para cambiar de detector (p. ej. YuNet), mantén el contrato `detect(frame) -> List[FaceResult]` y `annotate(...)`.

---

## 5. `app/models_loader.py`

### 5.1 Contrato de predicción

```python
@dataclass
class FacePrediction:
    gender: str          # "Masculino" | "Femenino"
    age: str             # bucket legible
    gender_conf: float   # 0..1
    age_conf: float      # 0..1
    @property
    def confidence(self) -> float:  # media redondeada a 3 decimales
```

`as_dict()` devuelve `gender`, `age`, `gender_confidence`, `age_confidence`, `confidence`.

### 5.2 `OnnxBackend`

**Carga:** `cv2.dnn.readNetFromONNX(path)`; intenta leer `getUnconnectedOutLayersNames()`.

**Preprocesado:**

```python
blob = cv2.dnn.blobFromImage(
    face_bgr,
    scalefactor=1/127.5,
    size=(224, 224),
    mean=(127.5, 127.5, 127.5),
    swapRB=True,
    crop=False,
)
```

Equivale a RGB normalizado `x/127.5 - 1`.

`blobFromImage` entrega un tensor **NCHW** `[1,3,224,224]`. El ONNX exportado por
`_build_nchw_export_model` declara esa misma entrada y hace el `Transpose` a NHWC
dentro del grafo, por lo que **no hay que transponer en la app**.

**Forward:** `_forward()` ejecuta **una sola vez**; ante `cv2.error` loguea y devuelve `None` (sin reintentos). El pipeline interpreta `None` como predicción ausente.

**Mapeo de salidas (`_split_outputs`):**

1. Por nombre: contiene `"gender"` / `"age"`.
2. Por forma: `size == len(age_buckets)` → edad; `size in (1, 2)` → género.
3. Si no cuadra → `RuntimeError`.

**Género:**

- Vector de 2: si ya es probabilidad (suma ≈ 1, rango [0,1]) se usa; si no, softmax. `female_prob = probs[1]`.
- Escalar: si está en `[0,1]` se usa; si no, sigmoide. `female_prob = raw`.

**Edad:** softmax si no es vector de probabilidad; `argmax` → `age_buckets[index]`.

### 5.3 `CaffeBackend`

- 4 archivos requeridos; si falta alguno → `FileNotFoundError` con la lista.
- `readNetFromCaffe(proto, model)` para edad y género.
- Blob 227×227, `scalefactor=1.0`, media `(78.4263377603, 87.7689143744, 114.895847746)`, `swapRB=False`.
- `gender_net`: `argmax` de 2 → `("Masculino", "Femenino")`.
- `age_net`: `argmax` de 8 → `CAFFE_AGE_BUCKETS` fijos.
- **No usa `AGE_BUCKETS`**: los buckets de Caffe son parte del modelo, no configurables.

### 5.4 `ModelHandle`

```python
@dataclass
class ModelHandle:
    requested: str
    backend: Optional[BaseBackend] = None
    error: str = ""
    # name -> backend.name o "none"
    # loaded -> backend is not None
    # predict(roi) -> Optional[FacePrediction]
    # describe() -> dict
```

### 5.5 `load_model(settings)`

```text
requested == "utkface_onnx" → _try_onnx()
requested == "caffe"        → _try_caffe()
requested == "auto"         → _try_onnx(); si None → _try_caffe()
si backend None → ModelHandle(error="; ".join(errors))
```

Los errores se acumulan y se exponen en `/api/status` como `model_error`.

### 5.6 Añadir un backend nuevo

1. Subclasea `BaseBackend` con `name`, `predict()` y `describe()`.
2. Añade un `_try_<backend>(settings, errors)`.
3. Extiende `VALID_BACKENDS` en `config.py` y la rama de `load_model`.
4. Si el modelo devuelve clases distintas, mapea a `FacePrediction` (no cambies el contrato).

---

## 6. `app/main.py`

### 6.1 `VideoPipeline`

#### Estado interno

| Campo | Protección | Descripción |
|---|---|---|
| `_jpeg` | `_lock` | Último JPEG publicado. |
| `_frame` | `_lock` | Último frame BGR (para snapshot). |
| `_faces` | `_lock` | Últimas caras detectadas. |
| `_fps` | `_lock` | FPS suavizado. |
| `_last_error` | `_lock` (mayoría) | Último error del worker. |
| `_stream_clients` | `_lock` | Clientes MJPEG activos. |
| `_thread`, `_stop`, `_wake`, `_started` | `_lifecycle_lock` | Ciclo de vida del worker. |

> `_last_error` se lee sin lock en la condición de limpieza (línea ~303) y `_started` en `status()`. Son lecturas de un `str`/`bool`; no hay corrupción, pero si endureces el código, protégelas también.

#### `start()` / `stop()`

- `start()` es idempotente: si el hilo vive, retorna. Si no, crea **nuevos** `Event` (`_stop`, `_wake`) y lanza `video-pipeline` daemon.
- `stop(join_timeout=3.0)` setea `_started=False`, dispara `_stop` y `_wake`, hace join acotado y limpia `_thread`.
- Los `Event` nuevos por ejecución evitan que un stop/start rápido reviva al worker anterior.

#### `_loop(stop, wake)`

```text
period = 1 / max(1, fps_target)
every_n = max(1, infer_every_n)
prev_ts = monotonic()

while not stop:
    try:
        if stream_clients <= 0:
            _idle_step()               # lee cámara, sin Haar/red/JPEG
            wake.wait(0.25); wake.clear()
            prev_ts = monotonic()
            continue

        frame = camera.read()
        if frame is None:
            si no hay jpeg o pasó >1 s → placeholder + encode + publish
            fps = 0
            wake.wait(0.25); wake.clear()
            continue

        limpiar "Error en pipeline" si se recuperó
        frame_index += 1
        si detector y (frame_index % every_n == 0 o no hay caras):
            faces = detector.detect(frame)   # try/except → error y []
            _faces = faces
        faces = copia de _faces
        annotated = frame.copy()
        detector.annotate(annotated, faces, fps, camera_ok=True)
        jpeg = _encode(annotated)
        si jpeg → _publish(annotated, jpeg, faces)
        fps EMA: 0.85*old + 0.15*instant
        wait = period - elapsed; stop.wait(wait)
    except Exception as exc:
        LOG.exception(...)
        _last_error = f"Error en pipeline: {exc}"
        wake.wait(0.5); wake.clear()
```

Puntos clave:

- **Un solo hilo** hace todo: no hay contención de OpenCV DNN entre clientes.
- **Detección no cada frame** por defecto (`INFER_EVERY_N=5`), pero **si no hay caras** se detecta en el siguiente frame (recuperación rápida al aparecer una persona).
- **Placeholder** se regenera como máximo cada 1 s para no codificar en bucle.
- **EMA de FPS** evita saltos en la UI.
- **Pacing**: si la iteración tardó menos que `period`, duerme lo que falta; si tardó más, no duerme.

#### Modo idle (`_idle_step`)

Sin clientes MJPEG:

- `camera.read()` (una vez por ciclo, ~250 ms).
- Guarda `_frame` (para snapshot), vacía `_faces`, `_fps=0`, `_jpeg=None`.
- No ejecuta Haar, ni red, ni `cv2.imencode`.
- `add_client()` hace `_wake.set()` para reanudar el stream de inmediato.

#### Publicación y lectura

- `_publish(frame, jpeg, faces)` escribe bajo `_lock`.
- `latest_jpeg()` devuelve la referencia al `bytes` (inmutable) sin copia.
- `latest_frame()` devuelve `frame.copy()` (mutable).
- `status()` calcula `message` según detector, `_started`, `is_running`, `camera.running`, `camera_ok` y `model_loaded`.

#### Mensajes de estado (orden de precedencia)

1. `Detector no disponible: ...` (detector `None`).
2. `Pipeline detenido` (`_started=False`).
3. `Pipeline detenido inesperadamente` (hilo muerto con `_started=True`).
4. `Cámara detenida` (`camera.running=False`).
5. Error de cámara / `Cámara no disponible`.
6. `Modelo no cargado: modo solo detección`.
7. `OK`.

### 6.2 `create_app(settings)`

Construye cámara, modelo, detector y pipeline; registra todo en `app.state`. Monta `/static` si existe.

#### `lifespan`

```python
async with control_lock:
    camera.start()
    pipeline.start()
yield
finally:
    async with control_lock:
        await asyncio.to_thread(pipeline.stop)
        await asyncio.to_thread(camera.stop)
```

El orden importa: al arrancar, primero cámara y luego pipeline; al parar, primero pipeline y luego cámara.

### 6.3 Endpoints

| Handler | Notas de implementación |
|---|---|
| `index()` | `FileResponse(static/index.html)`; 500 si falta. |
| `api_status()` | `pipeline.status()`. |
| `api_config()` | `cfg.public_config()` (sin secretos). |
| `healthz()` | `{"ok": True}`; útil para probes. |
| `api_camera_start()` | Token → `control_lock` → si ya está activa responde idempotente → stop pipeline/cámara → start cámara → espera ≤1.5 s a `is_ok` → 503 si falla → `pipeline.start()`. |
| `api_camera_stop()` | Token → `control_lock` → stop pipeline → stop cámara. |
| `api_snapshot()` | Token → rate-limit atómico (`snapshot_lock`) → `ensure_snap_folder()` (400/503) → `latest_frame()` (503) → PIL RGB quality 92 → fallback a `latest_jpeg()` o `imencode` → `{ok, path}`. |
| `video_feed()` | `StreamingResponse` con generador que registra cliente, hace yield solo cuando cambia el JPEG y se desregistra en `finally`. |

#### Rate-limit del snapshot

```python
now = time.monotonic()
with snapshot_lock:
    elapsed = now - last_ts
    if last_ts > 0 and elapsed < interval:
        retry_after = ceil(interval - elapsed)
        raise HTTPException(429, headers={"Retry-After": str(retry_after)})
    last_ts = now
```

Es atómico: el timestamp se actualiza **antes** de cualquier trabajo de disco, así dos peticiones concurrentes no pasan ambas.

#### Generador MJPEG

```python
pipeline.add_client()
last = None
try:
    while True:
        jpeg = pipeline.latest_jpeg()
        if jpeg is not None and jpeg is not last:
            last = jpeg
            yield header + jpeg + b"\r\n"
        elif jpeg is None:
            await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(0.03)
finally:
    pipeline.remove_client()
```

- `jpeg is not last` compara identidad de `bytes`: si el worker no publicó nada nuevo, no se reenvía.
- El `finally` garantiza decrementar el contador si el navegador cierra la conexión.

### 6.4 Autenticación

`_check_api_token(cfg, request)`:

- Si `API_TOKEN` está vacío → no-op.
- Si está definido → exige header `X-Api-Token` y compara con `hmac.compare_digest`.
- Falla con 401 + `WWW-Authenticate: ApiToken realm="AnalizadorRostroUNOQ"`.

Solo protege **POST** de escritura. `GET /`, `/video`, `/api/status`, `/api/config` y `/healthz` son públicos.

### 6.5 Entrypoints

| Archivo | Modo | Detalle |
|---|---|---|
| `app/main.py` (`__main__`) | Standalone | Configura logging, `load_settings()` y `uvicorn.run(app, host, port)`. |
| `python/main.py` | App Lab | `uvicorn.run("app.main:app", host, port, workers=1, proxy_headers=True)`. |

`workers=1` es obligatorio: hay estado compartido (cámara, pipeline, modelo) en memoria.

---

## 7. Pipeline de vídeo (diagrama)

```text
            ┌──────────────────────────────┐
            │ Camera thread (camera-capture)│
            │  V4L2 read() cada frame       │
            └──────────────┬───────────────┘
                           │ copia del frame
                           ▼
            ┌──────────────────────────────┐
            │ VideoPipeline (1 worker)      │
            │                               │
            │  ¿clientes MJPEG?             │
            │    no → idle (250 ms)         │
            │    sí → frame = camera.read() │
            │         │                     │
            │         ├─ frame None → placeholder (1 s)
            │         │                     │
            │         ├─ frame_index % N == 0 o sin caras
            │         │     → Haar detect   │
            │         │     → model.predict │
            │         │     → _faces        │
            │         │                     │
            │         ├─ annotate(copia)    │
            │         ├─ imencode JPEG      │
            │         └─ _publish(jpeg)     │
            └──────────────┬───────────────┘
                           │ bytes JPEG (bajo lock)
                           ▼
            ┌──────────────────────────────┐
            │ /video generator (por cliente)│
            │  yield solo si cambió el JPEG │
            └──────────────────────────────┘

  POST /api/snapshot → latest_frame() → PIL/imencode → archivo
  GET  /api/status   → pipeline.status() (snapshot del estado)
```

---

## 8. Contrato de API

### 8.1 `GET /api/status`

```json
{
  "camera_ok": true,
  "model_backend": "utkface_onnx",
  "model_backend_requested": "auto",
  "model_loaded": true,
  "degraded": false,
  "fps": 12.3,
  "faces_count": 1,
  "resolution": [640, 480],
  "message": "OK",
  "model_error": "",
  "camera_error": "",
  "pipeline_error": "",
  "pipeline_running": true,
  "stream_clients": 1
}
```

### 8.2 `GET /api/config`

```json
{
  "width": 640,
  "height": 480,
  "fps_target": 15,
  "model_backend": "auto",
  "infer_every_n": 5,
  "conf_threshold": 0.6,
  "snap_enabled": false,
  "snapshot_min_interval": 2.0,
  "auth_enabled": false,
  "age_buckets": ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+"],
  "camera_control": true,
  "snapshot_endpoint": "/api/snapshot",
  "jpeg_quality": 80
}
```

### 8.3 `POST /api/camera/start`

- 200: `{ok, camera: {...}, pipeline_running: true}`.
- 503: no se pudo abrir en 1.5 s (`detail` con el error de cámara).
- 401: token requerido/incorrecto.

### 8.4 `POST /api/camera/stop`

- 200: `{ok, camera: {...}, pipeline_running: false}`.

### 8.5 `POST /api/snapshot`

- 200: `{ok: true, path: "..."}`.
- 400: `SNAP_FOLDER` inválido/fuera del proyecto.
- 401: token.
- 429: rate-limit (`Retry-After`).
- 500: error de escritura.
- 503: snapshot deshabilitado o sin frame.

### 8.6 `GET /video`

- `Content-Type: multipart/x-mixed-replace; boundary=frame`.
- Cada parte: `--frame\r\nContent-Type: image/jpeg\r\nContent-Length: N\r\n\r\n<jpeg>\r\n`.
- `Cache-Control: no-store`, `Pragma: no-cache`, `Connection: close`.

---

## 9. Frontend (`static/`)

Sin build step ni dependencias.

| Archivo | Contenido |
|---|---|
| `index.html` | Estructura semántica, badges, panel de estado, controles, aviso de privacidad. |
| `app.js` | IIFE con estado local; polling de `/api/status` cada 1.2 s; `fetch` con `X-Api-Token`; `sessionStorage` para el token; reintento del stream a los 3.5 s si falla. |
| `style.css` | Tema oscuro, variables CSS, grid responsive, estados de badges, toast. |

### Lógica de `app.js`

```text
init()
  ├── bindControls()          listeners de botones/input/stream
  ├── loadConfig()            /api/config → snap_enabled, auth_enabled, infer_every_n
  ├── attachStream()          src=/video?t=timestamp
  ├── refreshStatus()         /api/status → applyStatus()
  └── setInterval(refreshStatus, 1200)
```

`friendlyError()` traduce 401/429/503 a mensajes en español. `applyStatus()` actualiza badges, panel, overlay y deshabilita botones según `camera_ok`.

> `GET /video` no lleva token: si `API_TOKEN` está activo, el stream sigue siendo visible para cualquiera que alcance el puerto. Es una decisión de diseño (MJPEG en `<img>` no envía headers personalizados). Si necesitas protegerlo, usa un proxy inverso con auth o un query token firmado (ver §12).

---

## 10. Scripts

### 10.1 `scripts/setup_board.sh`

```bash
bash scripts/setup_board.sh [--with-model] [--with-models]
```

- Verifica `python3` (o `PYTHON_BIN`).
- Crea `.venv` si no existe (falla con instrucciones si falta `python3-venv`).
- Actualiza `pip setuptools wheel` e instala `requirements.txt`.
- Con `--with-model`, llama a `download_model.sh` (ONNX UTKFace v1.0.0).
- Con `--with-models`, llama a `download_fallback_models.sh` (Caffe).
- Copia `.env.example` → `.env` si no existe.
- Idempotente.

### 10.2 `scripts/run.sh`

```bash
./scripts/run.sh
```

- `cd` a la raíz.
- Usa `.venv/bin/python` si existe; si no, `python3`.
- `exec python -m app.main`.
- `PYTHONUNBUFFERED=1`.

### 10.3 `scripts/deploy_unoq.sh`

```bash
./scripts/deploy_unoq.sh [--adb|--ssh] [--setup] [--model] [--models]
```

Variables: `REMOTE_APP_DIR` (default `/home/arduino/ArduinoApps/AnalizadorRostroUNOQ`), `UNOQ_HOST`, `UNOQ_USER` (default `arduino`), `UNOQ_SSH_KEY`.

- `--adb`: exige `adb devices` con al menos un `device`; hace `mkdir -p` y `adb push` de `ITEMS`.
- `--ssh`: exige `UNOQ_HOST`; `rsync -az` con exclusiones (`.venv`, `venv`, `__pycache__`, `.git`, `snapshots`, `dataset`, `.env`, `*.pyc`) o `scp` por ítem si no hay rsync.
- `--setup`: ejecuta `setup_board.sh` en remoto.
- `--model`: ejecuta `download_model.sh` en el host antes de copiar; el ONNX viaja
  dentro de `models/` y queda disponible en la placa.
- `--models`: añade `--with-models` al setup remoto (Caffe).
- `auto`: intenta ADB y cae a SSH.

> `.env` **nunca** se copia: evita filtrar tokens. El setup remoto crea uno desde `.env.example`.

### 10.4 `scripts/download_model.sh`

```bash
bash scripts/download_model.sh [--force]
```

- Descarga `utkface_mobilenetv2.onnx` desde el release `v1.0.0`.
- Variables `MODEL_URL` y `MODEL_SHA256` para apuntar a otro release/espejo.
- Verifica tamaño mínimo (9,000,000 bytes) y SHA256 antes de instalar.
- Usa `.part` y solo renombra si la verificación pasa; `--force` redescarga.
- Si el archivo ya existe y el hash coincide, no descarga.

### 10.5 `scripts/download_fallback_models.sh`

```bash
bash scripts/download_fallback_models.sh [destino]
```

- Descarga 4 archivos desde `smahesh29/Gender-and-Age-Detection`.
- Prueba nombres alternativos (`age_deploy.prototxt` / `deploy_age.prototxt`).
- Salta archivos existentes.
- Valida `.caffemodel` > 1 MB; borra y falla si es una página de error.
- `curl` con reintentos y timeout.

---

## 11. Entrenamiento (`training/train_utkface.py`)

### 11.1 Funciones

| Función | Propósito |
|---|---|
| `parse_buckets(spec)` | Valida exactamente 8 buckets. |
| `_bucket_bounds(bucket)` | Parsea `"20-29"` o `"60+"` a `(min, max)`. |
| `bucket_for_age(age, buckets)` | Índice de bucket; fuera de rango → extremo. |
| `_clean_stem(path)` | Elimina sufijos `.chip`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`. |
| `parse_filename(path)` | Regex `^(\d{1,3})_(\d)_\d+_(.+)$` sobre el stem; devuelve `(age, gender)` o `None`. |
| `run_self_test(verbose)` | 9 casos, incluido `26_1_3_20170116174519429.jpg.chip.jpg`. |
| `collect_samples(dir, validate)` | Recorre imágenes, parsea y opcionalmente descarta corruptas (OpenCV en modo reducido). |
| `maybe_download(dir, slug)` | `kaggle datasets download ... --unzip`. |
| `split_dataset(...)` | Estratificado 80/10/10 con `scikit-learn`. |
| `bucketize_ages(ages, buckets)` | Mapea edad cruda 0–120 → índice de bucket (0–7). Única fuente de verdad. |
| `build_datasets(...)` | `tf.data` con augmentación solo en train; aplica `bucketize_ages` antes del one-hot. |
| `build_model(num_buckets, lr)` | MobileNetV2 congelada + 2 cabezas. |
| `compile_model(model, lr)` | Adam, losses y `loss_weights={"gender":1.0,"age":2.0}`. |
| `unfreeze_top(base, n)` | Fine-tuning de las últimas `n` capas. |
| `ensure_onnx_output_names(path)` | Renombra/crea salidas `gender` y `age`. |
| `_build_nchw_export_model(model)` | Envuelve el modelo Keras NHWC con entrada NCHW `(None,3,224,224)` y `Transpose` interno. |
| `validate_onnx_model(path)` | Valida entrada `[N,3,224,224]` y salidas `gender`/`age`; si hay `onnxruntime`, ejecuta una inferencia dummy. |
| `export_onnx(model, path)` | Exporta el wrapper NCHW con `tf2onnx` opset 13; valida el resultado. |
| `export_tflite(model, path)` | Exportación opcional. |
| `evaluate(...)` | Accuracy, classification report, matriz de confusión, `evaluation.json`. |
| `save_history(...)` | `history.json` + `training_history.png`. |
| `main(argv)` | Orquesta todo; `--self-test` sale antes de importar TensorFlow. |

### 11.2 Contrato de exportación

- Entrada: `tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input")` (**NCHW**).
  El modelo Keras interno sigue siendo NHWC; `_build_nchw_export_model` inserta un
  `tf.transpose(input, [0, 2, 3, 1])` antes de la red.
- Salidas: `gender` (1 valor) y `age` (8 valores).
- Opset 13. `validate_onnx_model` comprueba el contrato y, si `onnxruntime` está
  instalado, ejecuta una inferencia dummy comprobando formas y valores finitos.
- `ensure_onnx_output_names` identifica por forma (`last == 1` → gender; `last == 8` → age) y añade nodos `Identity` si hace falta.

### 11.3 CLI

```text
--dataset-dir data/utkface   --download   --kaggle-dataset jangedoo/utkface-new
--epochs 50                  --fine-tune-epochs 10   --fine-tune-layers 30
--batch-size 32              --learning-rate 1e-4    --patience 8
--seed 42                    --buckets "0-2,..."     --output-dir output
--onnx-name utkface_mobilenetv2.onnx   --tflite   --no-validate   --self-test
```

### 11.4 Artefactos

```text
output/
├── best_model.keras
├── utkface_mobilenetv2.keras
├── utkface_mobilenetv2.onnx      ← copiar a models/
├── evaluation.json
├── confusion_matrix.png
├── training_history.png
├── history.json
└── fine_tune/                    (si fine-tuning)
```

> Si usas Keras 3 y falla `tf2onnx`, exporta con `TF_USE_LEGACY_KERAS=1` (documentado en `training/README.md`).

### 11.5 Modelo publicado v1.0.0

- Release: `https://github.com/wanderc137/AnalizadorRostroUNOQ/releases/tag/v1.0.0`
- Asset: `utkface_mobilenetv2.onnx` (10,211,778 bytes).
- SHA256: `390ade0e1f237616aca4636976f3223f1b8396e4bb539c27a4b55848a30c1661`.
- Métricas de test (2,371 imágenes): género 84.56 %, edad exacta 48.17 %,
  edad ±1 bucket 83.8 %.
- Entrenamiento: GitHub Actions, 10 épocas, MobileNetV2 congelada, lr 3e-4.
- El `.onnx` no se versiona; se obtiene con `scripts/download_model.sh` o desde
  el release.

---

## 12. Guía de modificación y extensión

### 12.1 Cambiar parámetros de Haar sin tocar código

Actualmente `scale_factor`, `min_neighbors` y `min_size` están fijados en `_build_pipeline`. Para exponerlos:

1. Añade campos a `Settings` (`haar_scale_factor`, `haar_min_neighbors`, `haar_min_size`).
2. Léelos en `load_settings()` con `_env_int` / `_env_float`.
3. Pásalos a `FaceDetector(...)` en `_build_pipeline`.
4. Documenta en `.env.example`.

### 12.2 Sustituir Haar por YuNet (recomendado para perfiles)

YuNet es más robusto y rápido que Haar. Mantén el contrato:

```python
class YuNetDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.6):
        self._det = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), score_threshold
        )

    def detect(self, frame: np.ndarray) -> list[FaceResult]:
        h, w = frame.shape[:2]
        self._det.setInputSize((w, h))
        _, faces = self._det.detect(frame)
        results = []
        if faces is not None:
            for row in faces:
                x, y, bw, bh = row[:4].astype(int)
                roi = frame[max(0, y):y + bh, max(0, x):x + bw]
                pred = self.model.predict(roi) if self.model.loaded and roi.size else None
                results.append(FaceResult((x, y, bw, bh), pred))
        return results

    def annotate(self, frame, faces, fps=None, camera_ok=True):
        # reutiliza la lógica de FaceDetector o compón un mixin
        ...
```

Pasos:

1. Descarga `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) a `models/`.
2. Añade `YUNET_MODEL_PATH` a `Settings`.
3. Cambia `_build_pipeline` para instanciar el detector nuevo.
4. Añade un flag `DETECTOR=haar|yunet` para poder volver atrás.
5. Verifica que `FaceResult.box` sigue siendo `(x, y, w, h)`.

> YuNet devuelve 5 landmarks; puedes usarlos para alinear el rostro antes de clasificar y mejorar la precisión.

### 12.3 Añadir tracking entre frames

Problema actual: las cajas se recalculan y pueden "saltar". Solución simple:

1. Calcula IoU entre cajas del frame anterior y actual.
2. Asigna IDs con matching greedy (IoU > 0.3).
3. Guarda `id` en `FaceResult` y dibújalo en la etiqueta.
4. Suaviza la caja con media móvil exponencial.

No cambies el contrato de `detect()`: el tracking puede vivir en `VideoPipeline` antes de `annotate`.

### 12.4 Publicar resultados por MQTT

Añade una tarea asíncrona en `lifespan`:

```python
import json
import paho.mqtt.client as mqtt

async def mqtt_publisher(pipeline, stop_event):
    client = mqtt.Client()
    client.connect("192.168.1.10", 1883, 60)
    client.loop_start()
    try:
        while not stop_event.is_set():
            status = pipeline.status()
            client.publish("unoq/rostros", json.dumps(status))
            await asyncio.sleep(2)
    finally:
        client.loop_stop()
        client.disconnect()
```

Consideraciones:

- Añade `paho-mqtt` a `requirements.txt`.
- No publiques imágenes ni datos identificables por defecto.
- Si publicas `faces_count`, es telemetría agregada; si publicas edades, es dato biométrico.

### 12.5 Añadir un sensor o actuador

Esta app no usa GPIO del MCU. Tienes dos caminos:

| Camino | Cuándo | Cómo |
|---|---|---|
| **Linux (MPU)** | Sensor I2C/SPI/UART/USB accesible desde Debian. | Usa `smbus2`, `pyserial`, `gpiod` dentro de un módulo nuevo; expón un endpoint o publica en el pipeline. |
| **MCU (Zephyr)** | Necesitas tiempo real, bajo consumo o pines del header Arduino. | Escribe firmware Zephyr aparte, compílalo con `arduino-cli`/App Lab y comunica con el MPU por USB/serie o un protocolo. **No forma parte de este proyecto.** |

Si añades un LED de estado en el MCU (D13), necesitarás:

1. Sketch/firmware Zephyr que escuche comandos por serie.
2. Un cliente en Python que abra `/dev/ttyACM0` y envíe el estado.
3. Manejo de reconexión y permisos (`dialout`).

No lo hagas desde `app/main.py` sin un módulo dedicado: acoplarías el pipeline a un dispositivo que puede no existir.

### 12.6 Proteger el stream MJPEG

`<img src="/video">` no envía headers personalizados. Opciones:

1. **Proxy inverso** (Nginx/Caddy) con Basic Auth o mTLS delante de `:8000`.
2. **Token en query** validado en `/video` (menos seguro: queda en logs).
3. **Cookie de sesión** emitida por un login y validada en `/video` (requiere implementar login).

La opción 1 es la más limpia y no toca el código.

### 12.7 Ejecutar como servicio systemd

```ini
# /etc/systemd/system/analizador-rostro.service
[Unit]
Description=AnalizadorRostroUNOQ
After=network-online.target

[Service]
Type=simple
User=arduino
WorkingDirectory=/home/arduino/ArduinoApps/AnalizadorRostroUNOQ
ExecStart=/home/arduino/ArduinoApps/AnalizadorRostroUNOQ/.venv/bin/python -m app.main
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now analizador-rostro
sudo systemctl status analizador-rostro
```

> Requiere autorización para tocar el sistema. Verifica que el usuario tenga acceso a `/dev/video0`.

### 12.8 Añadir métricas

Idea: contador de frames procesados, latencia de inferencia, rostros por frame.

1. Añade campos a `VideoPipeline` (`_frames_total`, `_infer_ms`).
2. Mide con `time.perf_counter()` alrededor de `detector.detect`.
3. Expón en `status()` o en un endpoint `/api/metrics`.
4. No incluyas datos personales.

### 12.9 Cambiar los buckets de edad

- **ONNX:** cambia `AGE_BUCKETS` en `.env` **y** reentrena con los mismos 8 buckets. Deben coincidir en orden y cantidad.
- **Caffe:** no se puede; los buckets están fijados en `CAFFE_AGE_BUCKETS` y el modelo fue entrenado así.
- Si cambias el número de clases, `_split_outputs` fallará con un `RuntimeError` claro.

---

## 13. Verificación y pruebas

### 13.1 Verificación estática

```bash
python3 -m py_compile app/main.py app/camera.py app/config.py \
  app/detector.py app/models_loader.py python/main.py training/train_utkface.py
bash -n scripts/*.sh
node --check static/app.js
python3 training/train_utkface.py --self-test
```

### 13.2 Pruebas manuales sugeridas

| Caso | Cómo | Resultado esperado |
|---|---|---|
| Sin cámara | `CAMERA_INDEX=99` | `camera_ok=false`, placeholder, UI con aviso. |
| Sin modelo | Renombrar `models/fallback` temporalmente | `model_loaded=false`, cajas ámbar, etiqueta "modelo no cargado". |
| 0 rostros | Apuntar a una pared | `faces_count=0`, FPS estable. |
| 1 rostro | Persona de frente | Una caja con etiqueta. |
| Varios rostros | 2–3 personas | Varias cajas, contador correcto. |
| Perfil | Girar la cabeza | Puede no detectar (limitación conocida). |
| Snapshot deshabilitado | `SNAP_FOLDER=` | 503. |
| Snapshot habilitado | `SNAP_FOLDER=snapshots` | 200 + archivo. |
| Rate-limit | Dos POST seguidos | 429 con `Retry-After`. |
| Token | `API_TOKEN=...` sin header | 401. |
| Path traversal | `SNAP_FOLDER=../fuera` | 400 / warning. |
| Cliente MJPEG | Abrir 2 navegadores | `stream_clients=2`, mismo FPS. |
| Stop/start rápido | Alternar botones | Sin hilos huérfanos; cámara reabre. |

### 13.3 Qué **no** está verificado

- Ejecución real en la UNO Q (la placa no estuvo accesible).
- FPS y temperatura en QRB2210.
- Compatibilidad de App Lab con `app.yaml`.
- Permisos de `/dev/video0` en la imagen Debian concreta de la placa.

---

## 14. Zephyr RTOS y diferencias con Arduino AVR clásico

| Concepto | Arduino AVR clásico | Esta app (UNO Q, MPU Linux) |
|---|---|---|
| Modelo de programa | `setup()` + `loop()` | `create_app()` + `lifespan` + hilos/async |
| Ejecución | Bare metal / sin SO | Debian Linux, procesos e hilos |
| Tiempo | `delay()`, `millis()` | `time.monotonic()`, `threading.Event.wait()`, `asyncio.sleep()` |
| Concurrencia | Interrupciones | Hilos + locks + `asyncio` |
| Memoria | Estática, KB | Dinámica, GB |
| Red | Shields/Wi-Fi externos | Wi-Fi/Ethernet nativos + HTTP |
| Cámara | No | USB UVC / V4L2 |
| Inferencia | No viable | OpenCV DNN en CPU |
| Compilación | `arduino-cli compile` | `pip install` (no hay compilación de sketch) |
| Carga | `arduino-cli upload` (USB/ICSP) | `adb push` / `rsync` / `scp` |
| GPIO | `digitalWrite()` | No se usan; si se usaran, `gpiod`/`sysfs` o firmware MCU |
| ADC | 10 bits | 12 bits en el MCU; no usado |
| Voltaje | 5 V | 3.3 V en GPIO del MCU |
| Librerías | Arduino core | FastAPI, OpenCV, NumPy, Pillow |
| RTOS | No | Zephyr en el MCU (no usado por esta app) |

### Lo que **no** puedes asumir

- No hay `Serial.print`; usa `logging`.
- No hay `String`; usa `str`.
- No hay `delay()` bloqueante en el worker; usa `Event.wait()` para poder cancelar.
- No hay `millis()`; usa `monotonic()` (no se ve afectado por cambios de reloj).
- No hay interrupciones de GPIO; hay hilos y eventos.
- No hay `setup()` que se ejecute una vez por arranque del MCU; hay `lifespan` por arranque del proceso.
- No hay watchdog del MCU; el worker tiene `try/except` para no morir, pero **no** reinicia el proceso si se cuelga.

### Sobre el MCU STM32U585

- Tiene TrustZone y Zephyr, pero este proyecto no lo programa.
- El FQBN `arduino:zephyr:unoq` y `/dev/ttyACM0` no se usan.
- Si en el futuro quieres repartir carga (p. ej. parpadeo de LED o lectura de un sensor), el MCU es una opción, pero requerirá firmware y un protocolo MPU↔MCU.

---

## 15. Deuda técnica y mejoras sugeridas

| Área | Mejora | Impacto |
|---|---|---|
| Detección | Migrar a YuNet o SCRFD | Mejor recall en perfiles y luz difícil. |
| Tracking | IoU + IDs + suavizado | Etiquetas estables, menos parpadeo. |
| Modelo | Reentrenar con datos locales y balanceo | Menos sesgo, mejor precisión. |
| Calibración | Temperatura scaling / calibración por clase | Confianzas más honestas. |
| Seguridad | Auth en `/video` vía proxy | Stream no público. |
| Robustez | Watchdog de proceso (systemd) | Recuperación automática. |
| Métricas | Latencia de inferencia y frames descartados | Diagnóstico de rendimiento. |
| Testing | Tests unitarios de `parse_filename`, `_split_outputs`, rate-limit | Regresiones detectadas antes. |
| Config | Exponer parámetros Haar/YuNet por `.env` | Ajuste sin tocar código. |
| Despliegue | CI que ejecute `py_compile`, `bash -n`, `node --check`, self-test | Calidad continua. |

---

## 16. Referencias internas

| Documento | Contenido |
|---|---|
| [`../README.md`](../README.md) | Instalación, operación, API, problemas. |
| [`datasheet.md`](datasheet.md) | Manual de usuario. |
| [`BITACORA.md`](BITACORA.md) | Decisiones, descartes y lecciones. |
| [`../PLAN.md`](../PLAN.md) | Especificación y criterios de aceptación. |
| [`../AUDIT_REPORT.md`](../AUDIT_REPORT.md) | Auditoría estática (9 incidencias resueltas). |
| [`../models/README.md`](../models/README.md) | Contrato de modelos. |
| [`../training/README.md`](../training/README.md) | Entrenamiento UTKFace. |
