from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset_text_elder_chunk_cached import MPDDElderTextChunkCachedDataset, collate_batch
from models.depformer_text_chunk_coral_mpdd import MPDDDepFormerTextChunkCoral
from models.depformer_text_coral_mpdd import predict_binary_from_coral, predict_binary_ternary_from_coral
from train_score_loss_text_coral import resolve_project_path


DEFAULT_BINARY_DIR = PROJECT_ROOT / "checkpoint/binary"
DEFAULT_TERNARY_DIR = PROJECT_ROOT / "checkpoint/ternary"
DEFAULT_PHQ_CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoint/ternary/04_modelid21_best_model_2026-06-03-13.31.36.pth"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "predict"
DEFAULT_DATA_ROOT = PROJECT_ROOT / "Elder-test"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "Elder-test/cached_av_chunks_c4_t128_mfcc2600"
DEFAULT_PERSONALITY_NPY = PROJECT_ROOT / "Elder-trainval/descriptions_embeddings_with_ids.npy"
DEFAULT_TEXT_NPY = PROJECT_ROOT / "Elder-test/sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Elder test with final chunkfull 5-vote checkpoints.")
    parser.add_argument("--binary_dir", type=Path, default=DEFAULT_BINARY_DIR)
    parser.add_argument("--ternary_dir", type=Path, default=DEFAULT_TERNARY_DIR)
    parser.add_argument("--phq_checkpoint", type=Path, default=DEFAULT_PHQ_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--av_cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--personality_npy", type=Path, default=DEFAULT_PERSONALITY_NPY)
    parser.add_argument("--text_embedding_npy", type=Path, default=DEFAULT_TEXT_NPY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return resolve_project_path(path)


def discover_checkpoints(path: Path) -> list[Path]:
    checkpoints = sorted(resolve(path).glob("*.pth"))
    if len(checkpoints) != 5:
        raise ValueError(f"Expected exactly 5 checkpoints under {path}, found {len(checkpoints)}")
    return checkpoints


def infer_test_ids(cache_dir: Path) -> list[int]:
    cache_dir = resolve(cache_dir)
    ids = sorted(int(path.stem) for path in cache_dir.glob("*.pt") if path.stem.isdigit())
    if not ids:
        raise RuntimeError(f"Cannot infer test IDs from cache dir: {cache_dir}")
    return ids


def build_dataset(
    *,
    task: str,
    checkpoint: dict[str, Any],
    test_ids: list[int],
    args: argparse.Namespace,
) -> MPDDElderTextChunkCachedDataset:
    label_map = {person_id: 0 for person_id in test_ids}
    source_split_map = {person_id: "test" for person_id in test_ids}
    target_t = int(checkpoint.get("target_t", checkpoint.get("model_kwargs", {}).get("target_t", 128)))
    return MPDDElderTextChunkCachedDataset(
        data_root=args.data_root,
        label_map=label_map,
        source_split_map=source_split_map,
        subtrack=checkpoint.get("subtrack", "A-V+P"),
        task=task,
        audio_feature=checkpoint.get("audio_feature", "all_audio"),
        video_feature=checkpoint.get("video_feature", "all_video"),
        personality_npy=args.personality_npy,
        text_embedding_npy=args.text_embedding_npy,
        phq_map=None,
        target_t=target_t,
        av_cache_dir=args.av_cache_dir,
    )


def forward_model(model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return model(
        mfcc=batch["mfcc"].to(device) if "mfcc" in batch else None,
        opensmile=batch["opensmile"].to(device) if "opensmile" in batch else None,
        wav2vec=batch["wav2vec"].to(device) if "wav2vec" in batch else None,
        densenet=batch["densenet"].to(device) if "densenet" in batch else None,
        resnet=batch["resnet"].to(device) if "resnet" in batch else None,
        openface=batch["openface"].to(device) if "openface" in batch else None,
        gait=batch["gait"].to(device) if "gait" in batch else None,
        personality=batch["personality"].to(device),
        pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
        chunk_mask=batch["chunk_mask"].to(device) if "chunk_mask" in batch else None,
        text_segments=batch["text_segments"].to(device),
        text_mask=batch["text_mask"].to(device),
    )


def predict_one(
    checkpoint_path: Path,
    *,
    task: str,
    args: argparse.Namespace,
    test_ids: list[int],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(resolve(checkpoint_path), map_location="cpu", weights_only=False)
    if checkpoint.get("backbone") != "depformer_text_coral":
        raise ValueError(f"{checkpoint_path} is not a depformer_text_coral checkpoint")
    if checkpoint.get("task") != task:
        raise ValueError(f"{checkpoint_path} is task={checkpoint.get('task')!r}, expected {task!r}")

    dataset = build_dataset(task=task, checkpoint=checkpoint, test_ids=test_ids, args=args)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    model = MPDDDepFormerTextChunkCoral(**dict(checkpoint["model_kwargs"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    ids: list[int] = []
    preds: list[int] = []
    phq_values: list[float] = []
    with torch.no_grad():
        for batch in loader:
            outputs = forward_model(model, batch, device)
            logits = outputs["coral_logits"]
            if task == "binary":
                pred = predict_binary_from_coral(logits)
            elif task == "ternary":
                _binary_pred, pred = predict_binary_ternary_from_coral(logits)
            else:
                raise ValueError(task)
            ids.extend(batch["pid"].cpu().numpy().astype(int).tolist())
            preds.extend(pred.detach().cpu().numpy().astype(int).tolist())
            phq_values.extend(outputs["phq_pred"].detach().cpu().numpy().astype(float).tolist())

    order = np.argsort(np.asarray(ids))
    return {
        "checkpoint": str(resolve(checkpoint_path)),
        "ids": np.asarray(ids, dtype=int)[order],
        "pred": np.asarray(preds, dtype=int)[order],
        "phq_pred": np.asarray(phq_values, dtype=float)[order],
        "experiment_name": checkpoint.get("experiment_name", resolve(checkpoint_path).parent.name),
        "seed": checkpoint.get("seed", ""),
    }


def majority_vote(pred_matrix: np.ndarray) -> np.ndarray:
    voted: list[int] = []
    for col in pred_matrix.T:
        counts = Counter(int(value) for value in col)
        voted.append(sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0])
    return np.asarray(voted, dtype=int)


def write_submission_csv(path: Path, ids: np.ndarray, pred_col: str, preds: np.ndarray, phq: np.ndarray) -> None:
    rows = []
    for person_id, pred, phq_value in zip(ids, preds, phq):
        rows.append({"id": int(person_id), pred_col: int(pred), "phq9_pred": f"{float(phq_value):.6f}"})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", pred_col, "phq9_pred"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    args.binary_dir = resolve(args.binary_dir)
    args.ternary_dir = resolve(args.ternary_dir)
    args.phq_checkpoint = resolve(args.phq_checkpoint)
    args.output_dir = resolve(args.output_dir)
    args.data_root = resolve(args.data_root)
    args.av_cache_dir = resolve(args.av_cache_dir)
    args.personality_npy = resolve(args.personality_npy)
    args.text_embedding_npy = resolve(args.text_embedding_npy)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    test_ids = infer_test_ids(args.av_cache_dir)
    binary_checkpoints = discover_checkpoints(args.binary_dir)
    ternary_checkpoints = discover_checkpoints(args.ternary_dir)

    binary_predictions = [
        predict_one(path, task="binary", args=args, test_ids=test_ids, device=device)
        for path in binary_checkpoints
    ]
    ternary_predictions = [
        predict_one(path, task="ternary", args=args, test_ids=test_ids, device=device)
        for path in ternary_checkpoints
    ]
    phq_prediction = predict_one(args.phq_checkpoint, task="ternary", args=args, test_ids=test_ids, device=device)

    ids = binary_predictions[0]["ids"]
    for item in [*binary_predictions, *ternary_predictions, phq_prediction]:
        if not np.array_equal(ids, item["ids"]):
            raise RuntimeError(f"ID order mismatch for checkpoint: {item['checkpoint']}")

    binary_vote = majority_vote(np.stack([item["pred"] for item in binary_predictions], axis=0))
    ternary_vote = majority_vote(np.stack([item["pred"] for item in ternary_predictions], axis=0))
    phq_values = np.clip(phq_prediction["phq_pred"], 0.0, 27.0)

    write_submission_csv(args.output_dir / "binary.csv", ids, "binary_pred", binary_vote, phq_values)
    write_submission_csv(args.output_dir / "ternary.csv", ids, "ternary_pred", ternary_vote, phq_values)
    for stale_name in ("binary_individual_predictions.csv", "ternary_individual_predictions.csv"):
        stale_path = args.output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    meta_path = args.output_dir / "prediction_meta.json"
    if meta_path.exists():
        meta_path.unlink()

    print(f"Wrote: {args.output_dir / 'binary.csv'}")
    print(f"Wrote: {args.output_dir / 'ternary.csv'}")


if __name__ == "__main__":
    main()
