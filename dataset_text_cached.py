from __future__ import annotations

import ctypes
import os
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_nvidia_library_path() -> None:
    candidates = [
        "/home/yfhou/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib",
        "/home/yfhou/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cublas/lib",
        "/home/yfhou/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cudnn/lib",
        "/home/yfhou/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cuda_runtime/lib",
        "/home/yfhou/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cusparse/lib",
        "/home/yfhou/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cusolver/lib",
    ]
    existing = [path for path in candidates if Path(path).exists()]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(existing + ([current] if current else []))
    for library in (
        "/home/yfhou/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib/libnvJitLink.so.12",
        "/home/yfhou/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cusparse/lib/libcusparse.so.12",
    ):
        if Path(library).exists():
            try:
                ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


_ensure_nvidia_library_path()

import torch

from dataset import (
    PAIR_COUNT,
    TARGET_T,
    _load_feature_array,
    _resize,
    resolve_project_path,
)
from dataset_text import (
    MPDDElderTextDataset,
    _stack_text_pairs,
)


DEFAULT_AV_CACHE_DIR = "Elder-trainval/cached_av_features_t128"
AV_KEYS = ("mfcc", "opensmile", "wav2vec", "densenet", "resnet", "openface")


def _default_cache_dir(data_root: str | Path, target_t: int) -> Path:
    root = resolve_project_path(data_root)
    return root / f"cached_av_features_t{target_t}"


class MPDDElderTextCachedDataset(MPDDElderTextDataset):
    def __init__(
        self,
        *args: Any,
        av_cache_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if av_cache_dir is None:
            av_cache_dir = _default_cache_dir(self.data_root, self.target_t)
        self.av_cache_dir = resolve_project_path(av_cache_dir)

    def _cache_path(self, person_id: int) -> Path:
        return self.av_cache_dir / f"{person_id}.pt"

    def _load_cached_av(self, person_id: int) -> dict[str, torch.Tensor]:
        cache_path = self._cache_path(person_id)
        if not cache_path.is_file():
            raise FileNotFoundError(
                f"A/V cache not found for id={person_id}: {cache_path}. "
                "Run feature_extract/cache_av_features.py before cached training."
            )
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")
        result: dict[str, torch.Tensor] = {}
        for key in AV_KEYS:
            if key not in payload:
                raise KeyError(f"Missing key '{key}' in A/V cache: {cache_path}")
            result[key] = payload[key].float()
        if "pair_mask" not in payload:
            raise KeyError(f"Missing key 'pair_mask' in A/V cache: {cache_path}")
        result["pair_mask"] = payload["pair_mask"].float()
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample: dict[str, Any] = self.samples[index]
        person_id = int(sample["pid"])
        result: dict[str, torch.Tensor] = {
            "pid": torch.tensor(person_id, dtype=torch.long),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
        }

        if self.has_phq_target:
            result["phq9"] = torch.tensor(float(sample["phq9"]), dtype=torch.float32)

        if self.need_av:
            result.update(self._load_cached_av(person_id))

        if self.need_gait:
            gait_arr = _load_feature_array(sample["gait_file"], self.gait_dim_hint, max_dim=9)
            result["gait"] = _resize(gait_arr, self.target_t)

        personality = self.personality_map.get(person_id, np.zeros(1024, dtype=np.float32))
        result["personality"] = torch.from_numpy(personality.astype(np.float32))

        pair_indices = list(sample.get("pair_indices", list(range(1, PAIR_COUNT + 1))))
        text_segments, text_mask = _stack_text_pairs(
            person_id=person_id,
            pair_indices=pair_indices,
            text_embedding_map=self.text_embedding_map,
        )
        result["text_segments"] = text_segments
        result["text_mask"] = text_mask
        return result


def infer_input_dims(dataset: MPDDElderTextCachedDataset) -> dict[str, int]:
    sample = dataset[0]
    return {
        "audio_dim": int(sample["audio"].shape[-1]) if "audio" in sample else 0,
        "video_dim": int(sample["video"].shape[-1]) if "video" in sample else 0,
        "gait_dim": int(sample["gait"].shape[-1]) if "gait" in sample else 0,
        "mfcc_dim": int(sample["mfcc"].shape[-1]) if "mfcc" in sample else 0,
        "opensmile_dim": int(sample["opensmile"].shape[-1]) if "opensmile" in sample else 0,
        "wav2vec_dim": int(sample["wav2vec"].shape[-1]) if "wav2vec" in sample else 0,
        "densenet_dim": int(sample["densenet"].shape[-1]) if "densenet" in sample else 0,
        "resnet_dim": int(sample["resnet"].shape[-1]) if "resnet" in sample else 0,
        "openface_dim": int(sample["openface"].shape[-1]) if "openface" in sample else 0,
        "text_dim": int(sample["text_segments"].shape[-1]) if "text_segments" in sample else 0,
        "text_max_segments": int(sample["text_segments"].shape[-2]) if "text_segments" in sample else 0,
    }
