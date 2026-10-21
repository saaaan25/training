"""
Evaluación de robustez de checkpoints Wav2Vec2 sobre audio degradado.

Evalúa varios entrenamientos sobre los MISMOS archivos, en limpio y con degradaciones
aplicadas en memoria (el dataset no se modifica). Cada variante puede tener varias semillas
(directorios separados por coma, en el mismo orden de semillas en todas las variantes).
Compara cada variante con la primera (baseline): por semilla, McNemar exacto y bootstrap
pareado; y en conjunto, media y desviación entre semillas de la diferencia de accuracy.

    python robust_eval.py --dataset-dir ../datasets/semi_dataset \
        --runs semi=./semi_results,./runs/semi_seed1 aug=./augmented_results,./runs/aug_seed1 \
        --output-dir ./robust_val

Por defecto usa validación. Test solo con --allow-test, y no debe usarse para decidir nada.
"""

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from scipy.stats import binomtest
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from common import LABELS, NO_VIOLENCE, TARGET_SR, read_json, set_seed
from data import AudioDataset
from models import build_model, forward_logits
from pipeline import load_frozen_weights, validation_path

MODEL = "wav2vec"


def add_noise(wave, rng, snr_db):
    power = wave.pow(2).mean().clamp(min=1e-10)
    noise = torch.from_numpy(rng.standard_normal(wave.numel()).astype("float32"))
    noise_power = power / (10 ** (snr_db / 10))
    return wave + noise * torch.sqrt(noise_power / noise.pow(2).mean())


def lowpass(wave, cutoff):
    return torchaudio.functional.lowpass_biquad(wave, TARGET_SR, cutoff)


def clip(wave, gain=4.0):
    return (wave * gain).clamp(-1.0, 1.0)


def reverb(wave, rng, rt60=0.6, length_s=0.5):
    """Reverberación sintética: respuesta al impulso de ruido con decaimiento exponencial."""
    n = int(length_s * TARGET_SR)
    t = torch.arange(n, dtype=torch.float32) / TARGET_SR
    ir = torch.from_numpy(rng.standard_normal(n).astype("float32")) * torch.exp(-6.9 * t / rt60)
    ir[0] = 1.0
    ir = ir / ir.abs().sum()
    size = wave.numel() + n - 1
    out = torch.fft.irfft(torch.fft.rfft(wave, size) * torch.fft.rfft(ir, size), size)
    return out[: wave.numel()]


CONDITIONS = {
    "clean": lambda w, r: w,
    "noise_20db": lambda w, r: add_noise(w, r, 20),
    "noise_10db": lambda w, r: add_noise(w, r, 10),
    "noise_5db": lambda w, r: add_noise(w, r, 5),
    "noise_0db": lambda w, r: add_noise(w, r, 0),
    "lowpass_2khz": lambda w, r: lowpass(w, 2000.0),
    "lowpass_1khz": lambda w, r: lowpass(w, 1000.0),
    "clipping": lambda w, r: clip(w),
    "reverb": lambda w, r: reverb(w, r),
}


class DegradedDataset(AudioDataset):
    """AudioDataset que degrada la forma de onda antes de la estandarización del modo raw.

    La semilla del ruido depende solo del nombre del archivo y de la condición, de modo que
    todos los modelos reciben exactamente el mismo audio degradado.
    """

    def __init__(self, dataset_dir, split, condition):
        super().__init__(dataset_dir, split, "raw")
        self.condition = condition

    def load_waveform(self, path):
        wave = AudioDataset.load_waveform(path)
        seed = zlib.crc32(f"{Path(path).name}|{self.condition}".encode())
        wave = CONDITIONS[self.condition](wave, np.random.default_rng(seed))
        peak = wave.abs().max()
        return wave / peak if peak > 0 else wave


@torch.no_grad()
def predict(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs, labels = [], []
    for inputs, targets in loader:
        logits = forward_logits(model, inputs.to(device), "raw")
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def summarize(probs, labels, threshold):
    pred = probs.argmax(axis=1)
    alert_true = labels != NO_VIOLENCE
    alert_pred = (1.0 - probs[:, NO_VIOLENCE]) >= threshold
    return {
        "accuracy": float((pred == labels).mean()),
        "f1_macro": float(f1_score(labels, pred, average="macro")),
        "alert_f1": float(f1_score(alert_true, alert_pred)),
        "alert_fpr": float(alert_pred[~alert_true].mean()) if (~alert_true).any() else float("nan"),
        "alert_fnr": float((~alert_pred[alert_true]).mean()),
    }


def paired_tests(correct_base, correct_other, iterations, seed):
    only_base = int((correct_base & ~correct_other).sum())
    only_other = int((~correct_base & correct_other).sum())
    discordant = only_base + only_other
    p_value = binomtest(only_other, discordant, 0.5).pvalue if discordant else 1.0
    rng = np.random.default_rng(seed)
    n = len(correct_base)
    delta = correct_other.astype(float) - correct_base.astype(float)
    diffs = np.array([delta[rng.integers(0, n, n)].mean() for _ in range(iterations)])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"only_baseline_correct": only_base, "only_other_correct": only_other,
            "mcnemar_p": float(p_value), "diff_acc": float(delta.mean()),
            "diff_ci95_low": float(low), "diff_ci95_high": float(high)}


def parse_runs(items):
    runs = []
    for item in items:
        name, _, paths = item.partition("=")
        if not paths:
            raise argparse.ArgumentTypeError(f"Use NOMBRE=DIR[,DIR...], recibido: {item}")
        runs.append((name, [Path(p) for p in paths.split(",")]))
    if len({n for n, _ in runs}) != len(runs):
        raise argparse.ArgumentTypeError("Nombres de variante repetidos.")
    if len({len(dirs) for _, dirs in runs}) != 1:
        raise argparse.ArgumentTypeError("Todas las variantes deben tener el mismo número de semillas.")
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Dataset cuya partición se evalúa; val/test deben ser idénticos en todos los runs.")
    parser.add_argument("--runs", nargs="+", required=True, metavar="NOMBRE=DIR[,DIR...]")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--conditions", nargs="+", choices=list(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.split == "test" and not args.allow_test:
        parser.error("Evaluar test requiere --allow-test (no úselo para tomar decisiones).")
    runs = parse_runs(args.runs)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results, correct, paths, train_seeds = [], {}, None, {}
    for name, run_dirs in runs:
        for index, run_dir in enumerate(run_dirs):
            record = read_json(validation_path(run_dir, MODEL))
            train_seed = record["training_arguments"]["seed"]
            train_seeds[(name, index)] = train_seed
            spec = record["training_arguments"]["wav2vec_settings"]["config"]["apply_spec_augment"]
            model, _ = build_model(MODEL, wav2vec_spec_augment=spec)
            load_frozen_weights(model, record, device)
            threshold = record["threshold"]["threshold"]

            for condition in args.conditions:
                dataset = DegradedDataset(args.dataset_dir, args.split, condition)
                if paths is None:
                    paths = dataset.relative_paths()
                elif paths != dataset.relative_paths():
                    raise ValueError("Los runs no evalúan los mismos archivos.")
                probs, labels = predict(model, dataset, device, args.batch_size)
                correct[(name, index, condition)] = probs.argmax(axis=1) == labels
                results.append({"run": name, "seed": train_seed, "condition": condition,
                                "n": len(labels), **summarize(probs, labels, threshold)})
                print(f"[{name} seed={train_seed}] {condition:13s} "
                      f"acc={results[-1]['accuracy']:.4f} f1={results[-1]['f1_macro']:.4f}", flush=True)
                pd.DataFrame({"relative_path": paths, "true": [LABELS[i] for i in labels],
                              **{f"prob_{l}": probs[:, i] for i, l in enumerate(LABELS)}}
                             ).to_csv(args.output_dir / f"pred_{name}_seed{train_seed}_{condition}.csv",
                                      index=False)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    table = pd.DataFrame(results)
    table.to_csv(args.output_dir / "robustness_metrics.csv", index=False)
    metrics = ["accuracy", "f1_macro", "alert_f1", "alert_fpr", "alert_fnr"]
    summary = (table.groupby(["run", "condition"])[metrics].agg(["mean", "std"])
               .reindex(args.conditions, level="condition"))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary["n_seeds"] = len(runs[0][1])
    summary.to_csv(args.output_dir / "robustness_summary.csv", index=False)

    base_name, base_dirs = runs[0]
    paired, per_seed = [], []
    for name, _ in runs[1:]:
        for condition in args.conditions:
            diffs = []
            for index in range(len(base_dirs)):
                if train_seeds[(base_name, index)] != train_seeds[(name, index)]:
                    raise ValueError("Las semillas de baseline y variante no coinciden en el orden dado.")
                test = paired_tests(correct[(base_name, index, condition)],
                                    correct[(name, index, condition)],
                                    args.bootstrap_iterations, args.seed)
                per_seed.append({"baseline": base_name, "other": name, "condition": condition,
                                 "seed": train_seeds[(name, index)], **test})
                diffs.append(test["diff_acc"])
            diffs = np.array(diffs)
            paired.append({"baseline": base_name, "other": name, "condition": condition,
                           "n_seeds": len(diffs), "mean_diff_acc": diffs.mean(),
                           "std_diff_acc": diffs.std(ddof=1) if len(diffs) > 1 else float("nan"),
                           "seeds_other_better": int((diffs > 0).sum()),
                           "seeds_tie": int((diffs == 0).sum()),
                           "seeds_other_worse": int((diffs < 0).sum())})
    if paired:
        pd.DataFrame(per_seed).to_csv(args.output_dir / "paired_tests_per_seed.csv", index=False)
        pd.DataFrame(paired).to_csv(args.output_dir / "paired_summary.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nAccuracy por condición (media entre semillas):")
    print(table.groupby(["condition", "run"]).accuracy.mean().unstack()
          .loc[args.conditions].to_string(float_format="%.4f"))
    if paired:
        print(f"\nDiferencia de accuracy contra '{base_name}' (otra - baseline), por semilla emparejada:")
        print(pd.DataFrame(paired).drop(columns="baseline").to_string(index=False, float_format="%.4f"))
    print(f"\nArchivos en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
