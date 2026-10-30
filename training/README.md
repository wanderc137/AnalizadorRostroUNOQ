# Entrenamiento UTKFace — género y rango de edad

Script `train_utkface.py`: entrena MobileNetV2 con dos cabezas sobre UTKFace y
exporta un ONNX (opset 13) listo para `models/utkface_mobilenetv2.onnx`.

> Este entrenamiento **no se ejecuta en la Arduino UNO Q**. Usa Google Colab
> (GPU gratuita) o un PC con GPU/CPU moderna.

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

## 2. Entorno en Colab

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

## 3. Pipeline del script

1. Parseo tolerante de nombres (`timestamp` de 13–17 dígitos, sufijos
   `.chip.jpg`) y filtrado de imágenes corruptas (OpenCV en modo reducido).
2. Bucketización de edad a las **8 clases fijas** (deben coincidir con
   `AGE_BUCKETS` de `app/config.py`):

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
9. Exportación ONNX opset 13 con salidas `gender` y `age`.

## 4. Contrato de inferencia (crítico)

- **Entrada**: RGB 224×224, normalizada `x / 127.5 - 1` (equivalente a
  `blobFromImage(face, 1/127.5, (224,224), (127.5,127.5,127.5), swapRB=True)`).
- **Salida `gender`**: sigmoide de 1 valor → `0` = Masculino, `1` = Femenino.
- **Salida `age`**: softmax de 8 valores en el orden de `AGE_BUCKETS`.

Si cambias los buckets aquí, actualiza `AGE_BUCKETS` en `.env` /
`app/config.py`; si no, las etiquetas dejarán de tener sentido.

## 5. Llevar el modelo a la UNO Q

```bash
# Desde el host de desarrollo
cp output/utkface_mobilenetv2.onnx ../models/utkface_mobilenetv2.onnx
./scripts/deploy_unoq.sh --models       # copia proyecto + modelo por ADB
```

En la placa:

```bash
cd /home/arduino/ArduinoApps/AnalizadorRostroUNOQ
MODEL_BACKEND=utkface_onnx ./scripts/run.sh
```

## 6. Notas de rendimiento

- El MPU QRB2210 no tiene GPU usable por OpenCV DNN: la inferencia ONNX corre
  en CPU. A 224×224 MobileNetV2 tarda decenas de ms por rostro.
- Para mantener 10–15 FPS de UI, sube `INFER_EVERY_N` (por ejemplo 8–10) o baja
  la resolución de captura a 480×360.
- La detección Haar corre en cada frame de inferencia; es barata comparada con
  la red.
