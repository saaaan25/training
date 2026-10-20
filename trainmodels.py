"""
Entrenamiento y evaluación final de clasificadores de violencia en audio.

Entrenar (todos los modelos o solo algunos):
    python trainmodels.py --phase train --dataset-dir ../../dataset/semi_dataset  --output-dir ./semi_results  --epochs 10
    python trainmodels.py --phase train --dataset-dir ../../dataset/final_dataset --output-dir ./final_results --epochs 10 --models wav2vec

Evaluar en test (una sola vez por modelo, tras cerrar las decisiones con validación):
    python trainmodels.py --phase test --dataset-dir ../../dataset/final_dataset --output-dir ./final_results

Modelos: mobilenet_v3, resnet50, densenet121, whisper, wav2vec, ast.
El metadata por defecto es <dataset-dir>/metadata.csv; use --metadata-path para otro archivo.
"""

import argparse
from pathlib import Path

import torch

from audit import audit_dataset
from common import MODELS
from pipeline import run_test, run_train


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=["train", "test"], required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--source-group-column", default="source_group",
                        help="Si falta, se usa source_parent.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=None,
                        help="Por defecto: todos (train) o todos los ya entrenados (test).")
    parser.add_argument("--best-only", action="store_true",
                        help="En test: evalúa solo el mejor modelo según validación.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--expected-per-class", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"),
                        help="Conteo esperado por clase. Se infiere en semi_dataset y final_dataset.")
    parser.add_argument("--allow-test-rerun", action="store_true",
                        help="Repite un test ya evaluado (solo recuperación técnica).")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("epochs y batch-size deben ser positivos; num-workers no negativo.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    expected = dict(zip(("train", "val", "test"), args.expected_per_class or ()))
    audit = audit_dataset(args.dataset_dir, args.metadata_path, args.source_group_column, expected)
    print(f"Auditoría completada ({audit['records']} registros). Huella: {audit['dataset_fingerprint'][:16]}...")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (run_train if args.phase == "train" else run_test)(args, device, audit)


if __name__ == "__main__":
    main()
