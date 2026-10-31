# Entrenamiento UTKFace — género y rango de edad

Script `train_utkface.py`: entrena MobileNetV2 con dos cabezas sobre UTKFace y
exporta un ONNX (opset 13) listo para `models/utkface_mobilenetv2.onnx`.

> Este entrenamiento **no se ejecuta en la Arduino UNO Q**. Usa GitHub Actions
> (CPU gratuita), Google Colab (GPU gratuita) o un PC con GPU/CPU moderna.

## 0. Modelo ya entrenado (v1.0.0)

No necesitas reentrenar para usar la app. El modelo está publicado en el release
`v1.0.0` y se descarga con verificación de SHA256:

```bash
./scripts/download_model.sh
```

- **Asset:** `utkface_mobilenetv2.onnx` (10.211.778 bytes).
- **SHA256:** `390ade0e1f237616aca4636976f3223f1b8396e4bb539c27a4b55848a30c1661`.
- **Test UTKFace (2.371 imágenes):** género 84.56 %, edad exacta 48.17 %,
  edad ±1 bucket 83.8 %.
- **Entrada:** NCHW `[N,3,224,224]`; salidas `gender` (1) y `age` (8).

Las secciones siguientes documentan cómo se generó y cómo reentrenarlo.

## 1. Dataset

- **Nombre**: UTKFace (Large Scale Face Dataset).
- **Kaggle**: `jangedoo/utkface-new` (o `jangedoo/utkface-new` → carpeta
  `UTKFace/` con archivos `[edad]_[genero]_[raza]_[fecha].jpg`).
- **Etiquetas del nombre**:
  - `edad`: entero 0–116.
  - `genero`: `0` = masculino, `1` = femenino.
  - `raza`: 0–4 (no se usa).
- **Atribución**: Zhang, Z., Song, Y., & Qi, H. (2017). *Age Progression /
  Regression in the Wild (UTKFace)*. ICCV Workshops. Uso académico; revisa la
  licencia del dataset antes de redistribuir imágenes.

### Formato de nombre aceptado

El parser es tolerante a las variantes reales del dataset:

- Patrón base: `^(\d{1,3})_(\d)_\d+_(.+)$` aplicado sobre el **stem**
  (`Path.stem`), no sobre el nombre completo.
- Timestamp de 13–17 dígitos (cualquier longitud, gracias a `\d+`).
- Sufijos tipo `.chip.jpg` o `.jpg.chip.jpg`: se eliminan iterativamente
  (`.chip`, `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`) antes del regex.

Ejemplos válidos:

```text
26_1_3_20170116174519429.jpg              -> edad=26, genero=1
26_1_3_20170116174519429.jpg.chip.jpg     -> edad=26, genero=1
1_0_0_20161219140623097.jpg               -> edad=1,  genero=0
```

Para verificar el parser sin TensorFlow ni dataset:

```bash
python train_utkface.py --self-test
```

### Descarga automática

```bash
# Necesita ~/.kaggle/kaggle.json (Account -> Create New API Token)
python train_utkface.py --download --dataset-dir data/utkface --epochs 50
```

### Descarga manual

```bash
kaggle datasets download -d jangedoo/utkface-new -p data --unzip
```

## 2. Entrenamiento en GitHub Actions (sin instalar en tu PC)

El workflow `.github/workflows/train-utkface.yml` entrena en un runner Ubuntu
de GitHub Actions, descarga UTKFace desde Hugging Face, exporta el ONNX y lo
publica como artifact. No requiere GPU ni TensorFlow local.

1. En GitHub: **Actions → Entrenar UTKFace (genero + edad) → Run workflow**.
2. Parámetros recomendados para CPU (6 h máximo): `epochs=10`,
   `batch_size=64`, `fine_tune_epochs=0`.
3. Al terminar, descarga el artifact `utkface-onnx` y copia
   `utkface_mobilenetv2.onnx` a `models/`, o publica un release con
   `gh release create`.
4. Verifica el SHA256 y el tamaño antes de desplegar a la UNO Q.

> Ejecución de referencia: run `41117352948` (commit `5ced197`), 10 épocas,
> ~37 min, **SUCCESS**. Género 84.56 %, edad exacta 48.17 %, ±1 bucket 83.8 %.

También se puede lanzar por CLI:

```bash
gh workflow run train-utkface.yml -f epochs=10 -f batch_size=64 -f fine_tune_epochs=0
gh run watch
gh run download --name utkface-onnx
```

El workflow valida el ONNX con `onnxruntime` y con `cv2.dnn` (entrada NCHW
`1×3×224×224`, salidas `gender` y `age`). El dataset **no** se sube al repo:
se descarga en el runner por su licencia de uso académico.

## 3. Entorno en Colab

```python
!pip install -r requirements-train.txt
# Sube train_utkface.py + requirements-train.txt, o clona el repo.
!python train_utkface.py --download --epochs 50 --batch-size 32 --output-dir output
```

Si usas Keras 3 y la exportación ONNX falla, ejecuta antes:

```python
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
```

## 4. Pipeline del script

1. Parseo tolerante de nombres (`timestamp` de 13–17 dígitos, sufijos
   `.chip.jpg`) y filtrado de imágenes corruptas (OpenCV en modo reducido).
2. Bucketización de edad a las **8 clases fijas** (deben coincidir con
   `AGE_BUCKETS` de `app/config.py`). La edad cruda 0-120 se mapea primero a
   un índice 0-7 con `bucketize_ages()`; recién entonces se aplica `one_hot`.
   Aplicar `one_hot` sobre la edad cruda produce vectores todo-cero para
   edades ≥ 8 y arruina el entrenamiento:

   `0-2`, `3-9`, `10-19`, `20-29`, `30-39`, `40-49`, `50-59`, `60+`

3. Split estratificado 80 / 10 / 10 (train / val / test).
4. Augmentación (solo train): rotación ±20°, zoom 15 %, traslación 10 %,
   flip horizontal y brillo ±20 %.
5. `MobileNetV2(include_top=False, weights="imagenet")` congelada + dos cabezas:
   - `gender`: Dense(128 relu) → Dropout(0.5) → Dense(1, sigmoid),
     `binary_crossentropy`.
   - `age`: Dense(128 relu) → Dropout(0.5) → Dense(8, softmax),
     `categorical_crossentropy`.
   - `loss_weights={"gender": 1.0, "age": 2.0}`, Adam `1e-4`.
6. Callbacks: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` (`.keras`).
7. Fine-tuning opcional de las últimas 30 capas con `lr/10`
   (`--fine-tune-epochs`, `--fine-tune-layers`).
8. Métricas de test: accuracy por cabeza, `classification_report` y matriz de
   confusión de edad (`evaluation.json`, `confusion_matrix.png`).
9. Exportación ONNX opset 13 con entrada NCHW (`1×3×224×224`), `Transpose` interno a NHWC y salidas `gender` y `age`.

## 5. Contrato de inferencia (crítico)

- **Entrada**: RGB 224×224, normalizada `x / 127.5 - 1` (equivalente a
  `blobFromImage(face, 1/127.5, (224,224), (127.5,127.5,127.5), swapRB=True)`).
- **Layout ONNX**: NCHW (`1×3×224×224`), exactamente el blob de OpenCV. El
  `Transpose` a NHWC queda dentro del ONNX; no cambies el preprocesado de la app.
- **Salida `gender`**: sigmoide de 1 valor → `0` = Masculino, `1` = Femenino.
- **Salida `age`**: softmax de 8 valores en el orden de `AGE_BUCKETS`.

Si cambias los buckets aquí, actualiza `AGE_BUCKETS` en `.env` /
`app/config.py`; si no, las etiquetas dejarán de tener sentido.

## 6. Llevar el modelo a la UNO Q

```bash
# Opción A: descargar el modelo publicado y desplegarlo
./scripts/download_model.sh
./scripts/deploy_unoq.sh --adb --setup --model

# Opción B: usar un modelo recién entrenado
cp training/output/utkface_mobilenetv2.onnx models/utkface_mobilenetv2.onnx
./scripts/deploy_unoq.sh --adb --setup
```

En la placa:

```bash
cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ
MODEL_BACKEND=utkface_onnx ./scripts/run.sh
```

`--model` descarga el ONNX en el host antes de copiar; `models/` viaja completo
en la transferencia. También puedes ejecutar en la placa
`bash scripts/setup_board.sh --with-model` si tiene internet.

## 7. Notas de rendimiento

- El MPU QRB2210 no tiene GPU usable por OpenCV DNN: la inferencia ONNX corre
  en CPU. A 224×224 MobileNetV2 tarda decenas de ms por rostro.
- Para mantener 10–15 FPS de UI, sube `INFER_EVERY_N` (por ejemplo 8–10) o baja
  la resolución de captura a 480×360.
- La detección Haar corre en cada frame de inferencia; es barata comparada con
  la red.
