"""Constantes y utilidades compartidas (sin dependencia de torch al importar)."""

import hashlib
import json
import logging
import os
import random
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

LABELS = ["no_violence", "physical_violence", "verbal_violence"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
NO_VIOLENCE = LABEL_TO_ID["no_violence"]
SPLITS = ("train", "val", "test")
MODELS = ("mobilenet_v3", "resnet50", "densenet121", "whisper", "wav2vec")

# Checkpoint de Whisper. Para migrar entre tamaños basta cambiar esta línea:
#   openai/whisper-tiny  (~8M params en el encoder)   openai/whisper-base  (~20M)
#   openai/whisper-small (~88M)                       openai/whisper-medium (~300M, no cabe en 4 GB)
# tiny/base/small/medium comparten el mismo extractor (80 mel bins); large-v3 usa 128.
# Al cambiar de tamaño: usar otro --output-dir (los checkpoints no son intercambiables) y revisar
# OTHER_LR["whisper"] en models.py (más grande = LR menor) y --batch-size si falta memoria.
WHISPER_CHECKPOINT = "openai/whisper-small"

TARGET_SR = 16000
FIXED_SAMPLES = TARGET_SR * 5


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Tipo no serializable en JSON: {type(value).__name__}")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False, default=_json_default)


def read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    import torch

    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


@contextmanager
def quiet():
    """Silencia stdout, stderr, warnings y logging durante una época."""
    previous = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with open(os.devnull, "w") as sink, warnings.catch_warnings(), \
                redirect_stdout(sink), redirect_stderr(sink):
            warnings.simplefilter("ignore")
            yield
    finally:
        logging.disable(previous)
