"""Construcción de modelos, configuración de wav2vec y optimizadores."""

from types import MethodType

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models as tv
from transformers import AutoModelForAudioClassification, WhisperForAudioClassification
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices

from common import LABELS, WHISPER_CHECKPOINT

NUM_CLASSES = len(LABELS)
TRANSFORMER_MODES = {"raw", "whisper"}

# Configuración de Wav2Vec2 Base (todo lo ajustable está aquí).
WAV2VEC_CONFIG = dict(
    apply_spec_augment=True,       # SpecAugment temporal
    mask_time_prob=0.05,
    mask_time_length=5,
    mask_time_min_masks=0,
    mask_feature_prob=0.0,         # solo máscara temporal
    layerdrop=0.0,                 # sin stochastic depth: pocas muestras y pocas épocas
    use_weighted_layer_sum=True,   # promedio ponderado de todas las capas (las intermedias rinden mejor que la última)
)
WAV2VEC_CHECKPOINT = "facebook/wav2vec2-base"
WAV2VEC_LR = {"encoder": 2e-5, "head": 2e-4}
WAV2VEC_HEAD = ("projector.", "classifier.", "layer_weights")
WAV2VEC_LABEL_SMOOTHING = 0.05
OTHER_LR = {"whisper": 5e-5}  # el resto de modelos usa 1e-3


def _masked_hidden_states(self, hidden_states, mask_time_indices=None, attention_mask=None):
    """SpecAugment temporal con torch.where (compatible con algoritmos deterministas en CUDA)."""
    if not self.config.apply_spec_augment:
        return hidden_states

    if mask_time_indices is None and self.training and self.config.mask_time_prob > 0:
        mask_time_indices = _compute_mask_indices(
            hidden_states.shape[:2],
            mask_prob=self.config.mask_time_prob,
            mask_length=self.config.mask_time_length,
            attention_mask=attention_mask,
            min_masks=self.config.mask_time_min_masks,
        )
    if mask_time_indices is not None:
        mask = torch.as_tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
        embedding = self.masked_spec_embed.to(hidden_states.dtype).view(1, 1, -1)
        hidden_states = torch.where(mask.unsqueeze(-1), embedding, hidden_states)
    return hidden_states


def _build_vision(name):
    if name == "mobilenet_v3":
        model = tv.mobilenet_v3_large(weights=tv.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    elif name == "resnet50":
        model = tv.resnet50(weights=tv.ResNet50_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        model = tv.densenet121(weights=tv.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
    return model


def build_model(name, wav2vec_spec_augment=None):
    """Devuelve (modelo, modo_de_entrada)."""
    labels = dict(
        num_labels=NUM_CLASSES,
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
        use_safetensors=True,
    )

    if name in {"mobilenet_v3", "resnet50", "densenet121"}:
        return _build_vision(name), "spec"

    if name == "wav2vec":
        config = dict(WAV2VEC_CONFIG)
        if wav2vec_spec_augment is not None:
            config["apply_spec_augment"] = wav2vec_spec_augment
        model = AutoModelForAudioClassification.from_pretrained(
            WAV2VEC_CHECKPOINT, **labels, **config
        )
        model.freeze_feature_encoder()
        if model.config.use_weighted_layer_sum:
            # Parámetro nuevo: from_pretrained lo deja sin inicializar (valores basura/NaN).
            with torch.no_grad():
                model.layer_weights.fill_(1.0 / model.layer_weights.numel())
        model.wav2vec2._mask_hidden_states = MethodType(_masked_hidden_states, model.wav2vec2)
        return model, "raw"

    if name == "whisper":
        return WhisperForAudioClassification.from_pretrained(WHISPER_CHECKPOINT, **labels), "whisper"

    raise ValueError(f"Modelo no reconocido: {name}")


def _wav2vec_logits(model, inputs):
    """inputs: (B, 2, N) con la forma de onda y su máscara (1 = muestra real, 0 = relleno).

    Mean-pooling solo sobre los frames reales para que el relleno no diluya la representación.
    """
    waveform, sample_mask = inputs[:, 0], inputs[:, 1]
    weighted = model.config.use_weighted_layer_sum
    attention_mask = sample_mask.long() if model.config.feat_extract_norm == "layer" else None
    outputs = model.wav2vec2(
        waveform, attention_mask=attention_mask, output_hidden_states=weighted
    )

    if weighted:
        stacked = torch.stack(outputs.hidden_states, dim=1)
        weights = torch.softmax(model.layer_weights, dim=-1)
        hidden = (stacked * weights.view(1, -1, 1, 1)).sum(dim=1)
    else:
        hidden = outputs.last_hidden_state
    hidden = model.projector(hidden)

    frames = model._get_feat_extract_output_lengths(sample_mask.sum(dim=1).long())
    if (frames <= 0).any():
        raise ValueError("Audio demasiado corto para producir frames v?lidos de Wav2Vec2.")
    frame_mask = (torch.arange(hidden.shape[1], device=hidden.device)[None, :]
                  < frames[:, None]).unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * frame_mask).sum(dim=1) / frame_mask.sum(dim=1).clamp(min=1)
    return model.classifier(pooled)


def forward_logits(model, inputs, mode):
    if mode == "raw":
        return _wav2vec_logits(model, inputs)
    if mode == "whisper":
        return model(input_features=inputs).logits
    return model(inputs)


def label_smoothing(name):
    return WAV2VEC_LABEL_SMOOTHING if name == "wav2vec" else 0.0


def build_optimizer(model, name):
    if name != "wav2vec":
        return optim.AdamW(model.parameters(), lr=OTHER_LR.get(name, 1e-3), weight_decay=1e-2)

    # Encoder con LR bajo, cabeza nueva con LR mayor; sin weight decay en bias/LayerNorm.
    groups = {}
    for pname, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lr = WAV2VEC_LR["head" if pname.startswith(WAV2VEC_HEAD) else "encoder"]
        decay = 0.0 if parameter.ndim <= 1 else 1e-2
        groups.setdefault((lr, decay), []).append(parameter)

    return optim.AdamW([{"params": p, "lr": lr, "weight_decay": decay}
                        for (lr, decay), p in groups.items()])
