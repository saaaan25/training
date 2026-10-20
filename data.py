"""Dataset de audio y DataLoaders."""

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from transformers import WhisperFeatureExtractor

from common import FIXED_SAMPLES, LABEL_TO_ID, LABELS, TARGET_SR, seed_worker

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class AudioDataset(Dataset):
    """mode: 'spec' (mel 3 canales para CNN), 'raw' (wav2vec) o 'whisper'."""

    def __init__(self, dataset_dir, split, mode):
        self.root = Path(dataset_dir)
        self.mode = mode
        self.items = [
            (path, LABEL_TO_ID[label])
            for label in LABELS
            for path in sorted((self.root / split / label).glob("*.wav"))
        ]
        if not self.items:
            raise RuntimeError(f"No se encontraron WAV en {self.root / split}")

        self.mel = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=TARGET_SR, n_fft=1024, hop_length=512, n_mels=128
            ),
            torchaudio.transforms.AmplitudeToDB(),
        )
        self.whisper = (WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")
                        if mode == "whisper" else None)

    def __len__(self):
        return len(self.items)

    def relative_paths(self):
        return [path.relative_to(self.root).as_posix() for path, _ in self.items]

    @staticmethod
    def load_waveform(path):
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[0] == 0 or not np.isfinite(audio).all():
            raise ValueError(f"Audio no válido: {path}")

        waveform = torch.from_numpy(audio.T.copy())
        if sample_rate != TARGET_SR:
            waveform = torchaudio.functional.resample(waveform, sample_rate, TARGET_SR)

        waveform = waveform.mean(dim=0)[:FIXED_SAMPLES]
        peak = waveform.abs().max()
        return waveform / peak if peak > 0 else waveform

    def __getitem__(self, index):
        path, label = self.items[index]
        waveform = self.load_waveform(path)

        if self.mode == "raw":
            # Normalización de wav2vec2 sobre las muestras reales, antes del relleno.
            waveform = (waveform - waveform.mean()) / torch.sqrt(
                waveform.var(unbiased=False) + 1e-7
            )
        waveform = F.pad(waveform, (0, FIXED_SAMPLES - waveform.numel()))

        if self.mode == "raw":
            features = waveform
        elif self.mode == "spec":
            mel = self.mel(waveform)
            mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-6)
            features = (mel.unsqueeze(0).repeat(3, 1, 1) - IMAGENET_MEAN) / IMAGENET_STD
        elif self.mode == "whisper":
            features = self.whisper(
                waveform.numpy(), sampling_rate=TARGET_SR, return_tensors="pt"
            ).input_features.squeeze(0)
        else:
            raise ValueError(f"Modo no soportado: {self.mode}")

        return features, torch.tensor(label, dtype=torch.long)


def make_loader(args, split, mode, shuffle):
    generator = torch.Generator().manual_seed(args.seed)
    return DataLoader(
        AudioDataset(args.dataset_dir, split, mode),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        generator=generator if shuffle else None,
        worker_init_fn=seed_worker if args.num_workers > 0 else None,
    )
