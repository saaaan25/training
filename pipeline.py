"""Fases de entrenamiento (train) y evaluación final (test)."""

import platform
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchaudio
import torchvision
from sklearn.metrics import confusion_matrix
from transformers import __version__ as transformers_version
from transformers import get_linear_schedule_with_warmup

from common import (LABELS, MODELS, NO_VIOLENCE, quiet, read_json, set_seed,
                    sha256_file, utc_now, write_json)
from data import make_loader
from metrics import (CLASS_IDS, binary_metrics, bootstrap_ci, choose_threshold,
                     class_report, epoch_metrics)
from models import (TRANSFORMER_MODES, build_model, build_optimizer, forward_logits,
                    label_smoothing)


def validation_path(out, name):
    return out / "validation" / f"{name}.json"


def test_dir(out, name):
    return out / "test" / name


def environment_versions():
    return {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "torchaudio": torchaudio.__version__,
        "torchvision": torchvision.__version__, "transformers": transformers_version,
        "numpy": np.__version__, "pandas": pd.__version__,
    }


def run_epoch(model, loader, device, mode, criterion, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, probabilities, labels = 0.0, [], []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.set_grad_enabled(training):
            logits = forward_logits(model, inputs, mode)
            loss = criterion(logits, targets)

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if mode in TRANSFORMER_MODES:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        probabilities.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
        labels.append(targets.cpu().numpy())

    probabilities, labels = np.concatenate(probabilities), np.concatenate(labels)
    metrics = epoch_metrics(total_loss / len(labels), labels, probabilities.argmax(axis=1))
    return metrics, probabilities, labels


def train_model(name, args, device, audit):
    out = args.output_dir
    set_seed(args.seed)

    model, mode = build_model(name)
    model.to(device)
    train_loader = make_loader(args, "train", mode, shuffle=True)
    val_loader = make_loader(args, "val", mode, shuffle=False)

    train_criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing(name))
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, name)

    scheduler = None
    if mode in TRANSFORMER_MODES:
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(total_steps * 0.10), num_training_steps=total_steps
        )

    weights_path = out / "checkpoints" / f"best_{name}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    best_key, best_epoch, best_val, history = None, None, None, []

    for epoch in range(1, args.epochs + 1):
        with quiet():
            train_m, _, _ = run_epoch(model, train_loader, device, mode, train_criterion,
                                      optimizer, scheduler)
            val_m, _, _ = run_epoch(model, val_loader, device, mode, eval_criterion)

        key = (val_m["f1_macro"], val_m["f1_weighted"], val_m["accuracy"], -val_m["loss"])
        if best_key is None or key > best_key:
            best_key, best_epoch, best_val = key, epoch, val_m
            torch.save(model.state_dict(), weights_path)

        history.append({
            "epoch": epoch,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        })
        print(f"[{name}] Época {epoch:02d}/{args.epochs:02d} | "
              f"Train macro-F1: {train_m['f1_macro']:.4f} | "
              f"Val macro-F1: {val_m['f1_macro']:.4f}", flush=True)

    history_path = out / "histories" / f"{name}_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_path, index=False)

    # Umbral de alerta con el mejor checkpoint, solo sobre validación.
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    _, val_probs, val_labels = run_epoch(model, val_loader, device, mode, eval_criterion)
    threshold, threshold_table = choose_threshold(val_probs, val_labels)
    (out / "validation").mkdir(parents=True, exist_ok=True)
    threshold_table.to_csv(out / "validation" / f"{name}_threshold_search.csv", index=False)

    write_json(validation_path(out, name), {
        "model": name,
        "input_mode": mode,
        "created_at_utc": utc_now(),
        "labels": LABELS,
        "dataset_fingerprint": audit["dataset_fingerprint"],
        "metadata_sha256": audit["metadata_sha256"],
        "best_epoch": best_epoch,
        "validation": best_val,
        "threshold": threshold,
        "weights_file": weights_path.resolve(),
        "weights_sha256": sha256_file(weights_path),
        "training_arguments": {
            "epochs": args.epochs, "batch_size": args.batch_size, "seed": args.seed,
            "label_smoothing": label_smoothing(name),
        },
        "environment": environment_versions(),
    })

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def leaderboard(out):
    rows = []
    for path in sorted((out / "validation").glob("*.json")):
        record = read_json(path)
        rows.append({
            "model": record["model"],
            "best_epoch": record["best_epoch"],
            **{f"val_{k}": v for k, v in record["validation"].items()},
            "alert_threshold": record["threshold"]["threshold"],
        })

    table = pd.DataFrame(rows).sort_values(
        ["val_f1_macro", "val_f1_weighted", "val_accuracy", "val_loss"],
        ascending=[False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    table.to_csv(out / "validation_leaderboard.csv", index=False)
    return table


def run_train(args, device, audit):
    out = args.output_dir
    names = args.models or list(MODELS)

    existing = [n for n in names
                if validation_path(out, n).exists() or (test_dir(out, n) / "completed.json").exists()]
    if existing:
        raise FileExistsError(
            f"Ya existe entrenamiento de {existing} en {out}. Use otro --output-dir "
            "o borre sus archivos si desea reentrenarlos."
        )

    write_json(out / "dataset_audit_train.json", audit)

    for name in names:
        print(f"\n>>> Entrenando {name}", flush=True)
        train_model(name, args, device, audit)

    print("\nRanking en validación (todos los modelos entrenados en este directorio):")
    print(leaderboard(out).to_string(index=False, float_format="%.4f"))
    print(f"\nEntrenamiento finalizado sin tocar test. Resultados en: {out.resolve()}")


def load_frozen_weights(model, record, device):
    if sha256_file(record["weights_file"]) != record["weights_sha256"]:
        raise ValueError("El checkpoint cambió desde el entrenamiento; se cancela la evaluación.")
    state = torch.load(record["weights_file"], map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()


def test_model(name, args, device, audit):
    out = args.output_dir
    record_file = validation_path(out, name)
    if not record_file.is_file():
        raise FileNotFoundError(
            f"{name} no tiene entrenamiento previo. Ejecute: --phase train --models {name}"
        )

    destination = test_dir(out, name)
    marker = destination / "completed.json"
    if marker.exists() and not args.allow_test_rerun:
        raise RuntimeError(
            f"Test de {name} ya evaluado. No reutilice test para tomar decisiones; "
            "use otro --output-dir o --allow-test-rerun solo para recuperación técnica."
        )

    record = read_json(record_file)
    if record["labels"] != LABELS:
        raise ValueError("El orden de etiquetas del entrenamiento no coincide.")
    if record["dataset_fingerprint"] != audit["dataset_fingerprint"]:
        raise ValueError("El corpus actual no coincide con el usado en el entrenamiento.")

    model, mode = build_model(name)
    load_frozen_weights(model, record, device)

    loader = make_loader(args, "test", mode, shuffle=False)
    metrics, probs, labels = run_epoch(model, loader, device, mode, nn.CrossEntropyLoss())
    predictions = probs.argmax(axis=1)
    threshold = record["threshold"]["threshold"]

    binary = binary_metrics(probs, labels, threshold)
    binary["confusion_matrix"] = [[binary["tn"], binary["fp"]], [binary["fn"], binary["tp"]]]
    report = class_report(labels, predictions)

    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(confusion_matrix(labels, predictions, labels=CLASS_IDS),
                 index=LABELS, columns=LABELS).to_csv(destination / "confusion_matrix.csv")
    report.to_csv(destination / "classification_report.csv", index=False)

    alert_probability = 1.0 - probs[:, NO_VIOLENCE]
    pd.DataFrame({
        "relative_path": loader.dataset.relative_paths(),
        "true_label": [LABELS[i] for i in labels],
        "predicted_label": [LABELS[i] for i in predictions],
        **{f"prob_{label}": probs[:, i] for i, label in enumerate(LABELS)},
        "alert_probability": alert_probability,
        "binary_alert_prediction": (alert_probability >= threshold).astype(int),
    }).to_csv(destination / "predictions.csv", index=False)

    write_json(destination / "final_metrics.json", {
        "evaluated_at_utc": utc_now(),
        "model": name,
        "weights_file": record["weights_file"],
        "weights_sha256": record["weights_sha256"],
        "labels": LABELS,
        "selected_threshold": record["threshold"],
        "test_multiclass_metrics": metrics,
        "test_class_metrics": report.to_dict(orient="records"),
        "test_binary_alert_metrics": binary,
        "bootstrap_ci_95": bootstrap_ci(labels, predictions, probs, threshold,
                                        args.bootstrap_iterations, args.seed),
        "dataset_fingerprint": audit["dataset_fingerprint"],
    })
    write_json(marker, {
        "completed_at_utc": utc_now(),
        "model": name,
        "weights_sha256": record["weights_sha256"],
        "selected_threshold": threshold,
        "allow_test_rerun_used": bool(args.allow_test_rerun),
        "note": "Test no debe reutilizarse para elegir arquitectura, épocas, umbral ni hiperparámetros.",
    })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_summary(out):
    rows = []
    for path in sorted((out / "test").glob("*/final_metrics.json")):
        result = read_json(path)
        multiclass, binary = result["test_multiclass_metrics"], result["test_binary_alert_metrics"]
        rows.append({
            "model": result["model"],
            "accuracy": multiclass["accuracy"],
            "f1_macro": multiclass["f1_macro"],
            "alert_precision": binary["precision"],
            "alert_recall": binary["recall"],
            "alert_f1": binary["f1"],
            "alert_fpr": binary["fpr"],
        })
    table = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    table.to_csv(out / "test_summary.csv", index=False)
    return table


def run_test(args, device, audit):
    out = args.output_dir
    trained = [n for n in MODELS if validation_path(out, n).is_file()]
    names = args.models or trained

    if args.best_only:
        ranking = leaderboard(out)["model"].tolist()
        names = [n for n in ranking if n in names][:1]
    if not names:
        raise FileNotFoundError(f"No hay modelos entrenados en {out}. Ejecute primero --phase train.")

    write_json(out / "dataset_audit_test.json", audit)

    for name in names:
        print(f"\n>>> Evaluando en test: {name}", flush=True)
        test_model(name, args, device, audit)

    print("\nResultados en test (todos los modelos evaluados en este directorio):")
    print(test_summary(out).to_string(index=False, float_format="%.4f"))
    print(f"\nDetalle por modelo en: {(out / 'test').resolve()}")
