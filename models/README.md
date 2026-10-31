# Modelos de género y edad

La aplicación soporta **dos backends** y elige según `MODEL_BACKEND`:

| `MODEL_BACKEND` | Backend usado | Archivos requeridos |
|---|---|---|
| `auto` (defecto) | ONNX propio, si no existe → Caffe, si no → solo detección | los de abajo |
| `utkface_onnx` | Solo modelo propio UTKFace | `models/utkface_mobilenetv2.onnx` |
| `caffe` | Solo fallback Caffe (Levi & Hassner) | `models/fallback/*` |

## 1. Modelo propio UTKFace (recomendado)

- **Ruta**: `models/utkface_mobilenetv2.onnx`
- **Formato**: ONNX opset 13, entrada `float32 [N, 3, 224, 224]` (**NCHW**).
- **Preprocesado**: BGR → RGB (`swapRB=True`), `(x - 127.5) / 127.5`, es decir
  `blobFromImage(face, 1/127.5, (224,224), (127.5,127.5,127.5), swapRB=True)`.
  El `Transpose` NCHW → NHWC está dentro del ONNX.
- **Salidas**:
  - `gender`: sigmoide de 1 valor → `0` = Masculino, `1` = Femenino.
  - `age`: softmax de 8 clases en el orden de `AGE_BUCKETS`:
    `0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60+`.
- **Modelo entrenado v1.0.0** (GitHub Release, no versionado en git):
  - URL: <https://github.com/wanderc137/AnalizadorRostroUNOQ/releases/download/v1.0.0/utkface_mobilenetv2.onnx>
  - SHA256: `390ade0e1f237616aca4636976f3223f1b8396e4bb539c27a4b55848a30c1661`
  - Tamaño: 10,211,778 bytes (aprox. 9.8 MB).
  - Métricas de test (UTKFace, 2,371 imágenes): género **84.56 %**;
    edad exacta **48.17 %**; edad ±1 bucket **83.8 %**.
- **Cómo obtenerlo**:

  ```bash
  ./scripts/download_model.sh          # descarga y verifica el SHA256
  ./scripts/download_model.sh --force  # vuelve a descargar
  ```

  También puedes entrenarlo con `training/train_utkface.py` y copiar el
  `.onnx` aquí. El ONNX **no se versiona** (ver `.gitignore`).

## 2. Fallback Caffe (Levi & Hassner)

```bash
./scripts/download_fallback_models.sh
```

Descarga en `models/fallback/`:

| Archivo | Tamaño esperado |
|---|---|
| `age_net.caffemodel` | > 1 MB |
| `gender_net.caffemodel` | > 1 MB |
| `deploy_age.prototxt` | pequeño (texto) |
| `deploy_gender.prototxt` | pequeño (texto) |

- **Entrada**: 227×227, media `(78.4263377603, 87.7689143744, 114.895847746)`,
  **sin** swapRB.
- **Género**: 2 clases → `0` = Masculino, `1` = Femenino.
- **Edad**: buckets clásicos del modelo, **distintos** a los de UTKFace:
  `0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100`
  (definidos en `app/models_loader.py`, no se reutiliza `AGE_BUCKETS`).

### Atribución Caffe

Levi, G., & Hassner, T. (2015). *Age and Gender Classification using
Convolutional Neural Networks*. CVPR Workshops.
Pesos publicados en `smahesh29/Gender-and-Age-Detection`; revisa la licencia del
repositorio original antes de redistribuirlos.

## 3. Modo degradado

Sin ningún modelo, la aplicación sigue funcionando:

- `/api/status` devuelve `model_loaded=false`, `degraded=true` y el motivo en
  `model_error`.
- El stream dibuja cajas Haar con la etiqueta
  `Rostro detectado | modelo no cargado`.
- La UI muestra el badge "Sin modelo" en rojo.

## 4. Verificación rápida

```bash
python - <<'PY'
import cv2
from app.config import load_settings
from app.models_loader import load_model
m = load_model(load_settings())
print(m.describe())
PY
```

## 5. Privacidad

Los modelos solo se cargan en memoria; no se envían imágenes a ningún servicio
externo. Los snapshots están desactivados salvo que definas `SNAP_FOLDER`.
