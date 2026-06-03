from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path
from typing import Any


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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (  # noqa: E402
    MPDDElderDataset,
    PAIR_COUNT,
    _load_feature_array,
    _resize,
    _resolve_feature_dir,
    _resolve_peer_feature_file,
    get_task_label,
    load_split_rows,
    resolve_project_path,
)


DEFAULT_DATA_ROOT = "Elder-trainval"
DEFAULT_SPLIT_CSV = "Elder-trainval/split_labels_train.csv"
DEFAULT_PERSONALITY_NPY = "Elder-trainval/descriptions_embeddings_with_ids.npy"
DEFAULT_OUTPUT_DIR = "Elder-trainval/cached_av_chunks_c4_t128_mfcc2600"
AV_KEYS = ("mfcc", "opensmile", "wav2vec", "densenet", "resnet", "openface")
FEATURE_SPECS = {
    "mfcc": ("audio", "mfcc", 64),
    "opensmile": ("audio", "opensmile", 65),
    "wav2vec": ("audio", "wav2vec", 768),
    "densenet": ("video", "densenet", 1000),
    "resnet": ("video", "resnet", 1000),
    "openface": ("video", "openface", 710),
}


def build_label_payload(split_csv: str | Path, task: str) -> tuple[dict[int, int], dict[int, str]]:
    rows = load_split_rows(split_csv)
    label_map: dict[int, int] = {}
    source_split_map: dict[int, str] = {}
    for row in rows:
        person_id = int(row["ID"])
        label_map[person_id] = get_task_label(row, task)
        source_split_map[person_id] = row.get("split", "train").strip().lower() or "train"
    return label_map, source_split_map


def infer_split_name(data_root: str | Path, split_name: str) -> str:
    if split_name != "auto":
        return split_name
    root_text = str(data_root).lower()
    return "test" if "test" in root_text else "train"


def infer_ids_from_feature_dirs(
    data_root: str | Path,
    audio_feature: str,
    video_feature: str,
    split_name: str,
) -> list[int]:
    root = resolve_project_path(data_root)
    search_audio = "mfcc" if audio_feature == "all_audio" else audio_feature
    search_video = "densenet" if video_feature == "all_video" else video_feature
    audio_root, _audio_alias = _resolve_feature_dir(root, "audio", split_name, search_audio)
    video_root, _video_alias = _resolve_feature_dir(root, "video", split_name, search_video)
    if not audio_root.exists():
        raise FileNotFoundError(f"Audio feature directory not found: {audio_root}")
    if not video_root.exists():
        raise FileNotFoundError(f"Video feature directory not found: {video_root}")

    audio_ids = {int(path.name) for path in audio_root.iterdir() if path.is_dir() and path.name.isdigit()}
    video_ids = {int(path.name) for path in video_root.iterdir() if path.is_dir() and path.name.isdigit()}
    ids = sorted(audio_ids & video_ids)
    if not ids:
        raise RuntimeError(f"No shared numeric subject IDs found under {audio_root} and {video_root}")
    return ids


def build_inferred_label_payload(
    data_root: str | Path,
    audio_feature: str,
    video_feature: str,
    split_name: str,
) -> tuple[dict[int, int], dict[int, str]]:
    inferred_split = infer_split_name(data_root, split_name)
    ids = infer_ids_from_feature_dirs(data_root, audio_feature, video_feature, inferred_split)
    return {person_id: 0 for person_id in ids}, {person_id: inferred_split for person_id in ids}


def parse_id_filter(value: str | None) -> set[int] | None:
    if value is None or value.strip() == "":
        return None
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def chunk_count_from_mfcc_frames(frame_count: int) -> int:
    if frame_count <= 2600:
        return 1
    if frame_count <= 5200:
        return 2
    if frame_count <= 7800:
        return 3
    return 4


def chunk_bounds(frame_count: int, chunk_idx: int, chunk_count: int) -> tuple[int, int]:
    start = int(round(frame_count * chunk_idx / chunk_count))
    end = int(round(frame_count * (chunk_idx + 1) / chunk_count))
    start = max(0, min(start, frame_count - 1))
    end = max(start + 1, min(end, frame_count))
    return start, end


def resize_chunks(array: Any, chunk_count: int, max_chunks: int, target_t: int, fallback_dim: int) -> torch.Tensor:
    frame_count = int(array.shape[0])
    chunks: list[torch.Tensor] = []
    for chunk_idx in range(chunk_count):
        start, end = chunk_bounds(frame_count, chunk_idx, chunk_count)
        chunks.append(_resize(array[start:end], target_t))
    while len(chunks) < max_chunks:
        chunks.append(torch.zeros(target_t, fallback_dim, dtype=torch.float32))
    return torch.stack(chunks[:max_chunks])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache Elder A/V features as 30s-style chunks per sample.")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--split_csv",
        default=DEFAULT_SPLIT_CSV,
        help="CSV with split/label/ID columns. Pass an empty string to infer IDs from feature folders.",
    )
    parser.add_argument(
        "--split_name",
        default="auto",
        choices=["auto", "train", "test"],
        help="Used only when --split_csv is empty. auto uses test if data_root contains 'test', otherwise train.",
    )
    parser.add_argument("--personality_npy", default=DEFAULT_PERSONALITY_NPY)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--task", default="ternary", choices=["binary", "ternary"])
    parser.add_argument("--subtrack", default="A-V+P", choices=["A-V+P"])
    parser.add_argument("--audio_feature", default="all_audio")
    parser.add_argument("--video_feature", default="all_video")
    parser.add_argument("--target_t", type=int, default=128)
    parser.add_argument("--max_chunks", type=int, default=4)
    parser.add_argument("--ids", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def cache_subject(
    dataset: MPDDElderDataset,
    sample: dict[str, Any],
    output_dir: Path,
    target_t: int,
    max_chunks: int,
    overwrite: bool,
) -> tuple[bool, Path]:
    person_id = int(sample["pid"])
    output_path = output_dir / f"{person_id}.pt"
    if output_path.exists() and not overwrite:
        return False, output_path

    source_split = sample["source_split"]
    audio_root = dataset.audio_roots[source_split]
    video_root = dataset.video_roots[source_split]
    audio_map: dict[int, Path] = sample["audio_map"]
    video_map: dict[int, Path] = sample["video_map"]
    pair_indices = list(sample.get("pair_indices", []))[:PAIR_COUNT]

    tensors: dict[str, list[torch.Tensor]] = {key: [] for key in AV_KEYS}
    pair_mask: list[float] = []
    chunk_mask_rows: list[torch.Tensor] = []
    chunk_counts: list[int] = []

    for pair_idx in pair_indices:
        audio_file = audio_map[pair_idx]
        video_file = video_map[pair_idx]
        mfcc_path = _resolve_peer_feature_file(audio_file, audio_root, "audio", "mfcc")
        mfcc = _load_feature_array(mfcc_path, fallback_dim=64)
        chunk_count = chunk_count_from_mfcc_frames(int(mfcc.shape[0]))

        for key in AV_KEYS:
            modality, feature_name, fallback_dim = FEATURE_SPECS[key]
            if modality == "audio":
                path = _resolve_peer_feature_file(audio_file, audio_root, "audio", feature_name)
            else:
                path = _resolve_peer_feature_file(video_file, video_root, "video", feature_name)
            array = mfcc if key == "mfcc" else _load_feature_array(path, fallback_dim=fallback_dim)
            tensors[key].append(resize_chunks(array, chunk_count, max_chunks, target_t, fallback_dim))

        mask = torch.zeros(max_chunks, dtype=torch.float32)
        mask[:chunk_count] = 1.0
        pair_mask.append(1.0)
        chunk_mask_rows.append(mask)
        chunk_counts.append(chunk_count)

    while len(pair_mask) < PAIR_COUNT:
        for key in AV_KEYS:
            _modality, _feature_name, fallback_dim = FEATURE_SPECS[key]
            tensors[key].append(torch.zeros(max_chunks, target_t, fallback_dim, dtype=torch.float32))
        pair_mask.append(0.0)
        chunk_mask_rows.append(torch.zeros(max_chunks, dtype=torch.float32))
        chunk_counts.append(0)

    payload: dict[str, Any] = {
        "id": person_id,
        "source_split": source_split,
        "pair_indices": pair_indices,
        "target_t": target_t,
        "max_chunks": max_chunks,
        "chunk_rule": "mfcc_frames<=2600:1,<=5200:2,<=7800:3,else:4",
        "pair_mask": torch.tensor(pair_mask, dtype=torch.float32),
        "chunk_mask": torch.stack(chunk_mask_rows),
        "chunk_count": torch.tensor(chunk_counts, dtype=torch.long),
    }
    for key in AV_KEYS:
        payload[key] = torch.stack(tensors[key]).contiguous()

    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)
    return True, output_path


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.split_csv).strip():
        label_map, source_split_map = build_label_payload(args.split_csv, args.task)
    else:
        label_map, source_split_map = build_inferred_label_payload(
            data_root=args.data_root,
            audio_feature=args.audio_feature,
            video_feature=args.video_feature,
            split_name=args.split_name,
        )
    id_filter = parse_id_filter(args.ids)
    if id_filter is not None:
        label_map = {person_id: label for person_id, label in label_map.items() if person_id in id_filter}
        source_split_map = {
            person_id: split_name
            for person_id, split_name in source_split_map.items()
            if person_id in id_filter
        }

    dataset = MPDDElderDataset(
        data_root=args.data_root,
        label_map=label_map,
        source_split_map=source_split_map,
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        phq_map=None,
        target_t=args.target_t,
    )

    total = len(dataset) if args.limit <= 0 else min(args.limit, len(dataset))
    written = 0
    skipped = 0
    for index in range(total):
        sample = dataset.samples[index]
        did_write, path = cache_subject(
            dataset=dataset,
            sample=sample,
            output_dir=output_dir,
            target_t=args.target_t,
            max_chunks=args.max_chunks,
            overwrite=args.overwrite,
        )
        if did_write:
            written += 1
            status = "wrote"
        else:
            skipped += 1
            status = "skip"
        print(f"[{index + 1}/{total}] {status} {int(sample['pid'])}: {path}", flush=True)

    print(f"Done. output_dir={output_dir} written={written} skipped={skipped} total={total}", flush=True)


if __name__ == "__main__":
    main()
