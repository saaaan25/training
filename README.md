# Clasificación de violencia en audio

Entrenamiento y evaluación de `mobilenet_v3`, `resnet50`, `densenet121`,
`whisper` y `wav2vec` para las clases `no_violence`, `physical_violence` y
`verbal_violence`. La selección del modelo y del umbral usa validación;
la evaluación final usa test por separado.

## Requisitos

- Python 3.14 de 64 bits para seguir esta guía.
- Windows o Linux. Los comandos siguientes incluyen ambos sistemas.
- Para GPU: una GPU NVIDIA y un controlador compatible con CUDA 12.8.
  En el equipo de origen se detectó una RTX 4050 de 6 GB con controlador 592.82.
- Internet para instalar dependencias y descargar los pesos preentrenados de
  Torchvision y Hugging Face en el primer uso, incluso al reconstruir modelos para test.
- El dataset y su `metadata.csv`, que se deben copiar por separado.

Combinación de paquetes para esta instalación:

| Paquete | Versión |
| --- | --- |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| torchaudio | 2.11.0 |
| Variante GPU | CUDA 12.8 (`cu128`) |

Esta combinación figura en las [instrucciones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/#v2110).
El resto de dependencias está en `requirements.txt`, sin versiones fijadas:
esta guía no constituye un bloqueo completo del entorno.

## 1. Copiar el proyecto y crear el entorno

Copia los archivos del proyecto al nuevo equipo y abre una terminal en esa carpeta.
No copies `.venv`: créalo de nuevo.

Windows, PowerShell:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

En los siguientes pasos, los comandos usan `python`. En Windows reemplázalo por
`.\.venv\Scripts\python.exe` para asegurar que se usa el entorno correcto, sin
necesidad de activarlo. En Linux mantén el entorno activado.

## 2. Instalar PyTorch y las dependencias

Elige **una** variante. Para GPU NVIDIA:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

Para trabajar exclusivamente en CPU, en Windows o Linux:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Después instala las demás dependencias, manteniendo la combinación elegida:

```bash
python -m pip install -r requirements.txt "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0"
python -m pip check
```

Si estás reparando un entorno que ya tenía otra variante de PyTorch, antes de
instalar ejecuta `python -m pip uninstall -y torch torchvision torchaudio`.

## 3. Verificar la instalación

```bash
python -c "import sys, torch, torchvision, torchaudio; print('Python:', sys.executable); print('Torch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('Torchaudio:', torchaudio.__version__); print('CUDA:', torch.version.cuda); print('GPU disponible:', torch.cuda.is_available())"
python trainmodels.py --help
```

Para GPU, `torch.version.cuda` debe mostrar `12.8` y la disponibilidad debe ser
`True`. Prueba también una operación real:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); x = torch.randn(512, 512, device='cuda'); y = x @ x; torch.cuda.synchronize(); print('Operacion correcta:', y.device)"
```

En VS Code selecciona el intérprete del entorno con **Python: Select Interpreter**:
`.venv\Scripts\python.exe` en Windows o `.venv/bin/python` en Linux.

## 4. Preparar el dataset

Cada dataset debe tener esta estructura; las tres carpetas de clases deben
existir en cada split:

```text
mi_dataset/
├── metadata.csv
├── train/
│   ├── no_violence/*.wav
│   ├── physical_violence/*.wav
│   └── verbal_violence/*.wav
├── val/
│   ├── no_violence/*.wav
│   ├── physical_violence/*.wav
│   └── verbal_violence/*.wav
└── test/
    ├── no_violence/*.wav
    ├── physical_violence/*.wav
    └── verbal_violence/*.wav
```

`metadata.csv` requiere `file_name`, `label`, `split`, `origin` y `source_group`
(admite `source_parent` como alternativa). Para audios aumentados también requiere
`parent_file` o `clean_parent` con un padre real de train de la misma clase y fuente.

- Los nombres de archivo deben ser únicos en todo el dataset.
- Una fuente no puede aparecer en varios splits.
- `val` y `test` solo admiten `origin=real_clean`.
- Los orígenes `augmented`, `augmented_existing` y `augmented_generated` solo se
  admiten en train.
- Para `semi_dataset` se esperan 400/50/50 archivos por clase en train/val/test;
  para `final_dataset`, 800/100/100. Para datasets con otro nombre, usa
  `--expected-per-class TRAIN VAL TEST` para especificar conteos por clase;
  esta opción no permite contradecir los perfiles de esos dos nombres.

El programa audita metadata, archivos, duplicados y separación de fuentes antes
de entrenar. Lee WAV, convierte a mono y 16 kHz, y recorta o rellena a 5 segundos.
Puedes indicar otra ubicación del CSV con `--metadata-path`.

## 5. Entrenar

Prueba inicial con un modelo y una época; reemplaza la ruta por la de tu dataset:

```bash
python trainmodels.py --phase train --dataset-dir ../../dataset/semi_dataset --output-dir ./smoke_results --epochs 1 --models mobilenet_v3 --batch-size 4
```

Entrenar todos los modelos:

```bash
python trainmodels.py --phase train --dataset-dir ../../dataset/semi_dataset --output-dir ./semi_results --epochs 10
```

Entrenar un modelo específico con el dataset final:

```bash
python trainmodels.py --phase train --dataset-dir ../../dataset/final_dataset --output-dir ./final_results --epochs 10 --models wav2vec --batch-size 4
```

Al iniciar verás `Dispositivo: cuda (...)` o `Dispositivo: cpu`.
`--models` acepta varios nombres separados por espacios.
Al especificar `--models`, la salida muestra únicamente los modelos indicados,
en el orden solicitado, sin ranking ni generación de resúmenes globales. En test,
`--best-only` restringe además la selección al mejor de los modelos indicados.
Si se omite `--models`, se conserva el resumen general del directorio.

Usa otro `--output-dir`
para repetir un experimento: el programa bloquea modelos ya entrenados en el mismo
directorio. La salida por época aparece al terminar entrenamiento y validación
de esa época; no hay barra de progreso por lote.

## 6. Evaluar en test

Después de cerrar las decisiones con validación, usa el mismo dataset y directorio
de resultados que en el entrenamiento:

```bash
python trainmodels.py --phase test --dataset-dir ../../dataset/final_dataset --output-dir ./final_results
```

Para evaluar solo el mejor según validación, añade `--best-only`. Para modelos
concretos usa `--models`. El test ya completado no se repite por defecto;
`--allow-test-rerun` está reservado a recuperación técnica.

## Resultados y traslado entre equipos

En cada directorio de resultados se guardan:

- `checkpoints/`: pesos del mejor checkpoint por modelo.
- `histories/`: métricas por época.
- `validation/` y `validation_leaderboard.csv`: selección del modelo y umbral.
- `dataset_audit_train.json` y `dataset_audit_test.json`: auditorías del corpus.
- `test/<modelo>/`: predicciones, reportes, matrices de confusión y métricas finales.
- `test_summary.csv`: resumen de evaluación.

`.gitignore` excluye entornos, cachés, datasets locales, pesos y directorios de
resultados habituales. Transfiere por separado el dataset y los resultados que
quieras conservar; usa un directorio terminado en `_results` para nuevas salidas.

Para evaluar en otro equipo un entrenamiento existente, copia **todo** el
directorio de resultados. Los JSON de `validation/` guardan `weights_file` como
**ruta absoluta**: actualiza ese campo a la ubicación del checkpoint en el nuevo
equipo. Conserva el checkpoint y `weights_sha256` sin cambios. La huella del
dataset también debe coincidir. Los JSON de auditoría y métricas pueden incluir
rutas del equipo original como información de procedencia.

Para conservar las versiones exactas después de verificar un entorno funcional:

```bash
python -m pip freeze > requirements-lock.txt
```

Guarda ese archivo junto al experimento. Para reconstruirlo, instala primero la
variante de PyTorch elegida como arriba y luego ejecuta
`python -m pip install -r requirements-lock.txt`.

## Problemas frecuentes

- **CUDA devuelve False:** comprueba `nvidia-smi`, el intérprete elegido y que
  `torch.__version__` tenga sufijo `+cu128` en la instalación GPU. Si termina en
  `+cpu`, reinstala los tres paquetes desde el índice CUDA indicado.
- **Error de DLL en Windows:** comprueba que ejecutas el Python de `.venv` y que
  los tres paquetes usan las versiones de la tabla, sin mezclar instalaciones.
- **Memoria GPU insuficiente:** reduce `--batch-size` a 2 o 1, especialmente con
  wav2vec. El consumo depende del modelo y del tamaño del lote.
- **CPU activa durante entrenamiento en GPU:** lectura, preparación del audio y
  métricas siguen usando CPU. `--num-workers` vale 0 por defecto; puedes probar
  2 para cargar datos en procesos separados.
- **Error de auditoría:** corrige metadata, conteos o separación de fuentes según
  el mensaje; cambiar el dispositivo no resuelve inconsistencias del corpus.
