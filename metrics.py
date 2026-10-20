"""Métricas multiclase, binarias (alerta / no alerta), umbral e intervalos bootstrap."""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from common import LABELS, NO_VIOLENCE

CLASS_IDS = list(range(len(LABELS)))


def epoch_metrics(loss, labels, predictions):
    return {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1_weighted": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro",
                                   labels=CLASS_IDS, zero_division=0)),
    }


def class_report(labels, predictions):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=CLASS_IDS, zero_division=0
    )
    return pd.DataFrame({"label": LABELS, "precision": precision, "recall": recall,
                         "f1_score": f1, "support": support})


def binary_metrics(probabilities, labels, threshold):
    """Alerta = cualquier clase distinta de no_violence."""
    truth = labels != NO_VIOLENCE
    predicted = (1.0 - probabilities[:, NO_VIOLENCE]) >= threshold

    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    tn = int((~predicted & ~truth).sum())

    def ratio(a, b):
        return a / b if b else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": ratio(tp + tn, len(truth)),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "fpr": ratio(fp, fp + tn),
        "fnr": ratio(fn, fn + tp),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def choose_threshold(probabilities, labels):
    """Umbral que maximiza el F1 binario en validación (desempate: recall, FPR, umbral)."""
    scores = 1.0 - probabilities[:, NO_VIOLENCE]
    thresholds = np.unique(np.concatenate([[0.0, 1.0], scores]))
    rows = [binary_metrics(probabilities, labels, t) for t in thresholds]

    selected = max(rows, key=lambda r: (r["f1"], r["recall"], -r["fpr"], r["threshold"])).copy()
    selected["selection_metric"] = "validation_binary_f1"
    return selected, pd.DataFrame(rows)


def bootstrap_ci(labels, predictions, probabilities, threshold, iterations, seed):
    """Bootstrap estratificado por clase, IC 95 %."""
    if iterations <= 0:
        return None

    rng = np.random.default_rng(seed)
    groups = [np.where(labels == c)[0] for c in CLASS_IDS]
    samples = {}

    for _ in range(iterations):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        y, pred = labels[idx], predictions[idx]

        row = {
            "multiclass_accuracy": accuracy_score(y, pred),
            "multiclass_f1_macro": f1_score(y, pred, average="macro",
                                            labels=CLASS_IDS, zero_division=0),
            "multiclass_f1_weighted": f1_score(y, pred, average="weighted", zero_division=0),
        }
        binary = binary_metrics(probabilities[idx], y, threshold)
        row.update({f"binary_alert_{k}": binary[k]
                    for k in ("precision", "recall", "f1", "fpr", "fnr")})

        for key, value in row.items():
            samples.setdefault(key, []).append(value)

    return {
        "method": "bootstrap_estratificado_por_clase",
        "iterations": int(iterations),
        "confidence_level": 0.95,
        "metrics": {
            key: {"low_95": float(np.percentile(v, 2.5)), "high_95": float(np.percentile(v, 97.5))}
            for key, v in samples.items()
        },
    }
