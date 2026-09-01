"""Dataset and dataloader construction for CAMoE experiments.

This module implements the data contract used by the training pipeline:
CMU-MOSI/CMU-MOSEI pickle files, optional modality-specific feature files,
aligned and unaligned metadata, and PyTorch dataloaders for all data splits.
It can coexist with the legacy loader and preserves its public entry points.
"""

from __future__ import annotations

import logging
import pickle
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


LOGGER = logging.getLogger("CAMoE")
SUPPORTED_DATASETS = frozenset({"mosi", "mosei"})
SUPPORTED_SPLITS = frozenset({"train", "valid", "test"})

__all__ = ["MMDataset", "MMDataLoader"]


def _arg_value(args: Any, name: str, default: Any = None) -> Any:
    """Read a configuration value from a mapping or attribute container."""

    if isinstance(args, Mapping):
        return args.get(name, default)
    return getattr(args, name, default)


def _set_arg_value(args: Any, name: str, value: Any) -> None:
    """Update a configuration value in a mapping or attribute container."""

    if isinstance(args, MutableMapping):
        args[name] = value
    else:
        setattr(args, name, value)


def _read_pickle(path: str | Path) -> Mapping[str, Any]:
    """Load a trusted preprocessed feature file."""

    feature_path = Path(path).expanduser()
    if not feature_path.is_file():
        raise FileNotFoundError(f"Feature file does not exist: {feature_path}")
    with feature_path.open("rb") as stream:
        content = pickle.load(stream)
    if not isinstance(content, Mapping):
        raise TypeError(f"Expected a mapping in feature file: {feature_path}")
    return content


def _float32(array: Any) -> np.ndarray:
    """Convert an array-like feature block to writable float32 storage."""

    return np.asarray(array, dtype=np.float32).copy()


class MMDataset(Dataset):
    """CMU-MOSI or CMU-MOSEI split backed by preprocessed NumPy arrays."""

    def __init__(self, args: Any, mode: str = "train") -> None:
        super().__init__()
        dataset_name = str(_arg_value(args, "dataset_name", "")).lower()
        if dataset_name not in SUPPORTED_DATASETS:
            choices = ", ".join(sorted(SUPPORTED_DATASETS))
            raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of: {choices}.")
        if mode not in SUPPORTED_SPLITS:
            choices = ", ".join(sorted(SUPPORTED_SPLITS))
            raise ValueError(f"Unsupported split '{mode}'. Expected one of: {choices}.")

        self.args = args
        self.mode = mode
        self.dataset_name = dataset_name
        self.use_bert = bool(_arg_value(args, "use_bert", False))
        self.is_aligned = bool(_arg_value(args, "need_data_aligned", False))

        source = _read_pickle(_arg_value(args, "featurePath"))
        self._initialize_from_source(source)

    def _initialize_from_source(self, source: Mapping[str, Any]) -> None:
        try:
            split = source[self.mode]
        except KeyError as exc:
            raise KeyError(f"Feature file does not contain the '{self.mode}' split.") from exc

        text_key = "text_bert" if self.use_bert else "text"
        self.text = _float32(split[text_key])
        self.audio = _float32(split["audio"])
        self.vision = _float32(split["vision"])
        self.raw_text = split["raw_text"]
        self.ids = split["id"]
        self.labels = {"M": _float32(split["regression_labels"])}

        audio_source = source
        vision_source = source
        text_override = _arg_value(self.args, "feature_T", "")
        audio_override = _arg_value(self.args, "feature_A", "")
        vision_override = _arg_value(self.args, "feature_V", "")

        if text_override:
            text_source = _read_pickle(text_override)
            text_split = text_source[self.mode]
            self.text = _float32(text_split[text_key])
            text_dim = 768 if self.use_bert else int(self.text.shape[-1])
            self._update_feature_dimension(0, text_dim)

        if audio_override:
            audio_source = _read_pickle(audio_override)
            self.audio = _float32(audio_source[self.mode]["audio"])
            self._update_feature_dimension(1, int(self.audio.shape[-1]))

        if vision_override:
            vision_source = _read_pickle(vision_override)
            self.vision = _float32(vision_source[self.mode]["vision"])
            self._update_feature_dimension(2, int(self.vision.shape[-1]))

        self._validate_sample_counts()

        if not self.is_aligned:
            self.audio_lengths = list(audio_source[self.mode]["audio_lengths"])
            self.vision_lengths = list(vision_source[self.mode]["vision_lengths"])

        # Retain the established behavior: invalid negative-infinity acoustic
        # values are treated as padded zeros.
        self.audio[np.isneginf(self.audio)] = 0.0

        if bool(_arg_value(self.args, "need_normalized", False)):
            self._normalize_temporal_features()

        LOGGER.info("%s samples: %s", self.mode, self.labels["M"].shape)

    def _update_feature_dimension(self, position: int, dimension: int) -> None:
        feature_dims = _arg_value(self.args, "feature_dims")
        if feature_dims is None:
            return
        if not isinstance(feature_dims, list):
            feature_dims = list(feature_dims)
            _set_arg_value(self.args, "feature_dims", feature_dims)
        feature_dims[position] = dimension

    def _validate_sample_counts(self) -> None:
        expected = len(self.labels["M"])
        arrays = {
            "text": self.text,
            "audio": self.audio,
            "vision": self.vision,
            "raw_text": self.raw_text,
            "id": self.ids,
        }
        mismatched = {name: len(value) for name, value in arrays.items() if len(value) != expected}
        if mismatched:
            details = ", ".join(f"{name}={count}" for name, count in mismatched.items())
            raise ValueError(f"Inconsistent sample counts; labels={expected}, {details}.")

    def _normalize_temporal_features(self) -> None:
        """Average acoustic and visual streams along their time dimension."""

        self.vision = self.vision.mean(axis=1, keepdims=True)
        self.audio = self.audio.mean(axis=1, keepdims=True)
        self.vision = np.nan_to_num(self.vision, nan=0.0)
        self.audio = np.nan_to_num(self.audio, nan=0.0)

    @staticmethod
    def _truncate_modality(features: np.ndarray, target_length: int) -> np.ndarray:
        """Extract a fixed window beginning at the first non-padding timestep."""

        if features.shape[1] == target_length:
            return features
        output = np.zeros(
            (features.shape[0], target_length, features.shape[2]),
            dtype=features.dtype,
        )
        for sample_index, sample in enumerate(features):
            non_padding = np.flatnonzero(np.any(sample != 0, axis=1))
            start = int(non_padding[0]) if non_padding.size else 0
            window = sample[start : start + target_length]
            output[sample_index, : len(window)] = window
        return output

    def _truncate(self) -> None:
        """Truncate each modality according to ``args.seq_lens`` when requested."""

        text_length, audio_length, vision_length = _arg_value(self.args, "seq_lens")
        if not self.use_bert:
            self.text = self._truncate_modality(self.text, int(text_length))
        self.audio = self._truncate_modality(self.audio, int(audio_length))
        self.vision = self._truncate_modality(self.vision, int(vision_length))

    def __len__(self) -> int:
        return len(self.labels["M"])

    def get_seq_len(self) -> tuple[int, int, int]:
        """Return text, acoustic, and visual sequence lengths."""

        text_length = self.text.shape[-1] if self.use_bert else self.text.shape[1]
        return int(text_length), int(self.audio.shape[1]), int(self.vision.shape[1])

    def get_feature_dim(self) -> tuple[int, int, int]:
        """Return configured text and observed acoustic/visual feature widths."""

        if self.use_bert:
            configured_dims = _arg_value(self.args, "feature_dims", [768])
            text_dimension = int(configured_dims[0])
        else:
            text_dimension = int(self.text.shape[-1])
        return text_dimension, int(self.audio.shape[-1]), int(self.vision.shape[-1])

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample: dict[str, Any] = {
            "raw_text": self.raw_text[index],
            "text": torch.from_numpy(self.text[index]),
            "audio": torch.from_numpy(self.audio[index]),
            "vision": torch.from_numpy(self.vision[index]),
            "index": index,
            "id": self.ids[index],
            "labels": {
                name: torch.from_numpy(values[index].reshape(-1))
                for name, values in self.labels.items()
            },
        }
        if not self.is_aligned:
            sample["audio_lengths"] = self.audio_lengths[index]
            sample["vision_lengths"] = self.vision_lengths[index]
        return sample


def MMDataLoader(args: Any, num_workers: int) -> dict[str, DataLoader]:
    """Build train, validation, and test dataloaders from one configuration."""

    datasets = {
        split: MMDataset(args, mode=split)
        for split in ("train", "valid", "test")
    }

    if (isinstance(args, Mapping) and "seq_lens" in args) or hasattr(args, "seq_lens"):
        _set_arg_value(args, "seq_lens", datasets["train"].get_seq_len())

    batch_size = int(_arg_value(args, "batch_size"))
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )
        for split, dataset in datasets.items()
    }
