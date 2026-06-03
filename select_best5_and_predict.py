from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from predict_chunkfull_5vote import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    DEFAULT_PERSONALITY_NPY,
    DEFAULT_TEXT_NPY,
    infer_test_ids,
    majority_vote,
    predict_one,
    resolve,
    write_submission_csv,
)

import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LOGS_ROOT = PROJECT_ROOT / "logs/cv5x20_chunkfull"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "predict_auto_selected"


@dataclass(frozen=True)
class Candidate:
    task: str
    result_json: str
    checkpoint: str
    experiment_name: str
    timestamp: str
    best_epoch: int
    selection_score: float
    ccc: float
    f1: float
    acc: float
    kappa: float
    binary_f1: float
    binary_acc: float
    binary_kappa: float
    ternary_f1: float
    ternary_acc: float
    ternary_kappa: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select top-5 binary/ternary checkpoints by validation selection_score, "
            "select one PHQ checkpoint by validation CCC, then predict Elder test."
        )
    )
    parser.add_argument("--logs_root", type=Path, default=DEFAULT_LOGS_ROOT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--av_cache_dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--personality_npy", type=Path, default=DEFAULT_PERSONALITY_NPY)
    parser.add_argument("--text_embedding_npy", type=Path, default=DEFAULT_TEXT_NPY)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument(
        "--phq_task",
        default="all",
        choices=["all", "binary", "ternary"],
        help="Candidate pool for selecting the PHQ-9 checkpoint by validation CCC.",
    )
    return parser


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def candidate_from_result(path: Path) -> Candidate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = str(payload.get("task", ""))
    if task not in {"binary", "ternary"}:
        raise ValueError(f"Unsupported task in {path}: {task!r}")

    metrics = payload.get("best_val_metrics") or {}
    checkpoint = resolve(payload["checkpoint_path"])
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint from {path} does not exist: {checkpoint}")

    return Candidate(
        task=task,
        result_json=str(path),
        checkpoint=str(checkpoint),
        experiment_name=str(payload.get("experiment_name", "")),
        timestamp=str(payload.get("timestamp", "")),
        best_epoch=as_int(payload.get("best_epoch")),
        selection_score=as_float(metrics.get("selection_score")),
        ccc=as_float(metrics.get("ccc")),
        f1=as_float(metrics.get("f1")),
        acc=as_float(metrics.get("acc")),
        kappa=as_float(metrics.get("kappa")),
        binary_f1=as_float(metrics.get("binary_f1")),
        binary_acc=as_float(metrics.get("binary_acc")),
        binary_kappa=as_float(metrics.get("binary_kappa")),
        ternary_f1=as_float(metrics.get("ternary_f1")),
        ternary_acc=as_float(metrics.get("ternary_acc")),
        ternary_kappa=as_float(metrics.get("ternary_kappa")),
    )


def load_candidates(logs_root: Path) -> list[Candidate]:
    logs_root = resolve(logs_root)
    paths = sorted(logs_root.glob("**/train_result_*.json"))
    if not paths:
        raise FileNotFoundError(f"No train_result_*.json found under {logs_root}")
    candidates: list[Candidate] = []
    for path in paths:
        candidates.append(candidate_from_result(path))
    return candidates


def select_top_by_score(candidates: list[Candidate], task: str, top_k: int) -> list[Candidate]:
    task_candidates = [item for item in candidates if item.task == task]
    if len(task_candidates) < top_k:
        raise RuntimeError(f"Need at least {top_k} {task} candidates, found {len(task_candidates)}")
    return sorted(
        task_candidates,
        key=lambda item: (item.selection_score, item.f1, item.kappa, item.ccc),
        reverse=True,
    )[:top_k]


def select_best_phq(candidates: list[Candidate], phq_task: str) -> Candidate:
    if phq_task == "all":
        pool = candidates
    else:
        pool = [item for item in candidates if item.task == phq_task]
    if not pool:
        raise RuntimeError(f"No candidates available for phq_task={phq_task!r}")
    return sorted(
        pool,
        key=lambda item: (item.ccc, item.selection_score, item.f1, item.kappa),
        reverse=True,
    )[0]


def assert_same_ids(predictions: list[dict[str, Any]]) -> np.ndarray:
    ids = predictions[0]["ids"]
    for item in predictions[1:]:
        if not np.array_equal(ids, item["ids"]):
            raise RuntimeError(f"ID order mismatch for checkpoint: {item['checkpoint']}")
    return ids


def main() -> None:
    args = build_parser().parse_args()
    args.logs_root = resolve(args.logs_root)
    args.output_dir = resolve(args.output_dir)
    args.data_root = resolve(args.data_root)
    args.av_cache_dir = resolve(args.av_cache_dir)
    args.personality_npy = resolve(args.personality_npy)
    args.text_embedding_npy = resolve(args.text_embedding_npy)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.logs_root)
    binary_selected = select_top_by_score(candidates, "binary", args.top_k)
    ternary_selected = select_top_by_score(candidates, "ternary", args.top_k)
    phq_selected = select_best_phq(candidates, args.phq_task)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    test_ids = infer_test_ids(args.av_cache_dir)

    binary_predictions = [
        predict_one(Path(item.checkpoint), task="binary", args=args, test_ids=test_ids, device=device)
        for item in binary_selected
    ]
    ternary_predictions = [
        predict_one(Path(item.checkpoint), task="ternary", args=args, test_ids=test_ids, device=device)
        for item in ternary_selected
    ]
    phq_prediction = predict_one(
        Path(phq_selected.checkpoint),
        task=phq_selected.task,
        args=args,
        test_ids=test_ids,
        device=device,
    )

    ids = assert_same_ids([*binary_predictions, *ternary_predictions, phq_prediction])
    binary_vote = majority_vote(np.stack([item["pred"] for item in binary_predictions], axis=0))
    ternary_vote = majority_vote(np.stack([item["pred"] for item in ternary_predictions], axis=0))
    phq_values = np.clip(phq_prediction["phq_pred"], 0.0, 27.0)

    write_submission_csv(args.output_dir / "binary.csv", ids, "binary_pred", binary_vote, phq_values)
    write_submission_csv(args.output_dir / "ternary.csv", ids, "ternary_pred", ternary_vote, phq_values)

    meta_path = args.output_dir / "prediction_meta.json"
    if meta_path.exists():
        meta_path.unlink()

    print("Selected binary checkpoints:")
    for idx, item in enumerate(binary_selected, start=1):
        print(f"  {idx}. score={item.selection_score:.6f} f1={item.f1:.6f} kappa={item.kappa:.6f} ccc={item.ccc:.6f} {item.checkpoint}")
    print("Selected ternary checkpoints:")
    for idx, item in enumerate(ternary_selected, start=1):
        print(f"  {idx}. score={item.selection_score:.6f} f1={item.f1:.6f} kappa={item.kappa:.6f} ccc={item.ccc:.6f} {item.checkpoint}")
    print(
        "Selected PHQ checkpoint: "
        f"task={phq_selected.task} ccc={phq_selected.ccc:.6f} "
        f"score={phq_selected.selection_score:.6f} {phq_selected.checkpoint}"
    )
    print(f"Wrote: {args.output_dir / 'binary.csv'}")
    print(f"Wrote: {args.output_dir / 'ternary.csv'}")


if __name__ == "__main__":
    main()
