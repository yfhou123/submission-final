from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import ctypes

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

from dataset import (  # re-export common dataset utilities for train_text.py later
    PAIR_COUNT,
    REGRESSION_TASK,
    TARGET_T,
    MPDDElderDataset,
    build_subset_label_map,
    collate_batch,
    load_label_maps,
    load_task_maps,
    resolve_project_path,
)


TEXT_MAX_SEGMENTS = 24
TEXT_EMBED_DIM = 1024
DEFAULT_TEXT_EMBEDDING_NPY = (
    "Elder-trainval/sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy"
)


def _default_text_embedding_path(data_root: str | Path) -> Path:
    root = resolve_project_path(data_root)
    if root.name == "Elder-trainval" or root.parent.name == "MPDD-AVG2026-trainval":
        return root / "sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy"
    if root.name == "Elder-test" or root.parent.name == "MPDD-AVG2026-test":
        return root / "sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy"
    return resolve_project_path(DEFAULT_TEXT_EMBEDDING_NPY)


def _load_text_embedding_map(text_embedding_npy: str | Path | None) -> dict[tuple[int, str], tuple[np.ndarray, np.ndarray]]:
    if text_embedding_npy is None:
        return {}
    path = resolve_project_path(text_embedding_npy)
    if not path.exists():
        return {}

    data = np.load(str(path), allow_pickle=True)
    text_map: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    for item in data:
        person_id = int(item["id"])
        sample = str(item["sample"])
        embedding = np.asarray(item["embedding"], dtype=np.float32)
        mask = np.asarray(item["mask"], dtype=np.float32)
        if embedding.ndim != 2:
            continue
        if mask.ndim != 1:
            continue
        text_map[(person_id, sample)] = (embedding, mask)
    return text_map


def _empty_text_embedding() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((TEXT_MAX_SEGMENTS, TEXT_EMBED_DIM), dtype=np.float32),
        np.zeros((TEXT_MAX_SEGMENTS,), dtype=np.float32),
    )


def _fit_text_shape(embedding: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fitted_embedding = np.zeros((TEXT_MAX_SEGMENTS, TEXT_EMBED_DIM), dtype=np.float32)
    fitted_mask = np.zeros((TEXT_MAX_SEGMENTS,), dtype=np.float32)
    seg_count = min(TEXT_MAX_SEGMENTS, embedding.shape[0], mask.shape[0])
    dim_count = min(TEXT_EMBED_DIM, embedding.shape[1])
    if seg_count > 0 and dim_count > 0:
        fitted_embedding[:seg_count, :dim_count] = embedding[:seg_count, :dim_count]
        fitted_mask[:seg_count] = mask[:seg_count]
    return fitted_embedding, fitted_mask


def _stack_text_pairs(
    person_id: int,
    pair_indices: list[int],
    text_embedding_map: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]],
) -> tuple[torch.Tensor, torch.Tensor]:
    segment_pairs: list[np.ndarray] = []
    mask_pairs: list[np.ndarray] = []
    for pair_idx in pair_indices[:PAIR_COUNT]:
        sample_name = f"A_{pair_idx}"
        embedding, mask = text_embedding_map.get((person_id, sample_name), _empty_text_embedding())
        embedding, mask = _fit_text_shape(embedding, mask)
        segment_pairs.append(embedding)
        mask_pairs.append(mask)

    while len(segment_pairs) < PAIR_COUNT:
        embedding, mask = _empty_text_embedding()
        segment_pairs.append(embedding)
        mask_pairs.append(mask)

    return (
        torch.from_numpy(np.stack(segment_pairs).astype(np.float32)),
        torch.from_numpy(np.stack(mask_pairs).astype(np.float32)),
    )


class MPDDElderTextDataset(MPDDElderDataset):
    def __init__(
        self,
        data_root: str | Path,
        label_map: dict[int, int],
        source_split_map: dict[int, str],
        subtrack: str,
        task: str,
        audio_feature: str,
        video_feature: str,
        personality_npy: str | Path,
        text_embedding_npy: str | Path | None = None,
        phq_map: dict[int, float] | None = None,
        target_t: int = TARGET_T,
    ) -> None:
        self.text_embedding_path = (
            resolve_project_path(text_embedding_npy)
            if text_embedding_npy is not None
            else _default_text_embedding_path(data_root)
        )
        self.text_embedding_map = _load_text_embedding_map(self.text_embedding_path)
        super().__init__(
            data_root=data_root,
            label_map=label_map,
            source_split_map=source_split_map,
            subtrack=subtrack,
            task=task,
            audio_feature=audio_feature,
            video_feature=video_feature,
            personality_npy=personality_npy,
            phq_map=phq_map,
            target_t=target_t,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = super().__getitem__(index)
        sample: dict[str, Any] = self.samples[index]
        person_id = int(sample["pid"])
        pair_indices = list(sample.get("pair_indices", list(range(1, PAIR_COUNT + 1))))
        text_segments, text_mask = _stack_text_pairs(
            person_id=person_id,
            pair_indices=pair_indices,
            text_embedding_map=self.text_embedding_map,
        )
        result["text_segments"] = text_segments
        result["text_mask"] = text_mask
        return result


def infer_input_dims(dataset: MPDDElderTextDataset) -> dict[str, int]:
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
