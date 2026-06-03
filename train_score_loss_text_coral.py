from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from dataset_text import (
    DEFAULT_TEXT_EMBEDDING_NPY,
    MPDDElderTextDataset,
    collate_batch,
    infer_input_dims,
    resolve_project_path,
)
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from metrics import classification_metrics, regression_metrics, safe_float
from models import MPDDDepFormerTextCoral
from models.depformer_text_coral_mpdd import PHQ_MAX, predict_binary_from_coral, predict_binary_ternary_from_coral
from models.heads_mpdd import build_phq_ordinal_targets
from train_val_split import create_train_val_split


PROJECT_ROOT = Path(__file__).resolve().parent
TASK_THRESHOLDS = {
    "binary": (5.0,),
    "ternary": (5.0, 10.0),
}
SUBTRACK_LOG_DIRS = {
    "A-V+P": "A-V-P",
    "A-V-G+P": "A-V-G+P",
    "G+P": "G-P",
}
METRIC_ARRAY_KEYS = {"ids", "y_true", "y_pred", "class_true", "class_pred", "phq_true", "phq_pred"}
PATH_ARG_KEYS = {
    "config",
    "data_root",
    "split_csv",
    "personality_npy",
    "text_embedding_npy",
    "checkpoints_dir",
    "logs_dir",
}


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(resolve_project_path(config_path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train task-specific DepFormer-text simplified CORAL model.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--track", default=defaults["track"], choices=["Track1", "Track2"])
    parser.add_argument("--task", default=defaults.get("task", "ternary"), choices=["binary", "ternary"])
    parser.add_argument("--subtrack", default="A-V+P", choices=["A-V+P"])
    parser.add_argument("--encoder_type", default=defaults["encoder_type"], choices=["bilstm_mean", "hybrid_attn"])
    parser.add_argument("--backbone", default="depformer_text_coral", choices=["depformer_text_coral"])
    parser.add_argument("--depformer_d_model", type=int, default=defaults.get("depformer_d_model", 256))
    parser.add_argument("--depformer_adapter_dim", type=int, default=defaults.get("depformer_adapter_dim", 128))
    parser.add_argument("--depformer_lstm_layers", type=int, default=defaults.get("depformer_lstm_layers", 1))
    parser.add_argument("--depformer_bct_layers", type=int, default=defaults.get("depformer_bct_layers", 1))
    parser.add_argument("--depformer_heads", type=int, default=defaults.get("depformer_heads", 2))
    parser.add_argument("--sample_pooling", default=defaults.get("sample_pooling", "mean"), choices=["mean", "attention"])
    parser.add_argument("--lambda_reg", type=float, default=defaults.get("lambda_reg", 1.0))
    parser.add_argument("--audio_feature", default=defaults.get("audio_feature", "all_audio"))
    parser.add_argument("--video_feature", default=defaults.get("video_feature", "all_video"))
    parser.add_argument("--data_root", default=defaults["data_root"])
    parser.add_argument("--split_csv", default=defaults["split_csv"])
    parser.add_argument("--personality_npy", default=defaults["personality_npy"])
    parser.add_argument("--text_embedding_npy", default=defaults.get("text_embedding_npy", DEFAULT_TEXT_EMBEDDING_NPY))
    parser.add_argument("--val_ratio", type=float, default=defaults["val_ratio"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument(
        "--fixed_val_ids_path",
        default=defaults.get(
            "fixed_val_ids_path",
            "splits/cv5_true_seed3407/track1_elder_ternary_cv5_seed3407_fold4.json",
        ),
    )
    parser.add_argument("--fixed_split_label", default=defaults.get("fixed_split_label", "label3"), choices=["label2", "label3", "task"])
    parser.add_argument("--epochs", type=int, default=defaults["epochs"])
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"])
    parser.add_argument("--lr", type=float, default=defaults["lr"])
    parser.add_argument("--weight_decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--target_t", type=int, default=defaults["target_t"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--hidden_dim", type=int, default=defaults["hidden_dim"])
    parser.add_argument("--dropout", type=float, default=defaults["dropout"])
    parser.add_argument("--patience", type=int, default=defaults["patience"])
    parser.add_argument("--min_delta", type=float, default=defaults["min_delta"])
    parser.add_argument("--num_workers", type=int, default=defaults["num_workers"])
    parser.add_argument("--checkpoints_dir", default=defaults["checkpoints_dir"])
    parser.add_argument("--logs_dir", default=defaults["logs_dir"])
    parser.add_argument("--experiment_name", default="")
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config", default="config.json")
    known_args, _ = base_parser.parse_known_args()
    defaults = load_config(known_args.config)
    return build_parser(defaults).parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"text_coral_{log_file.stem}_{time.time_ns()}")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def resolve_track_task_dir(root: Path, track: str, subtrack: str, task: str, experiment_name: str) -> Path:
    del track, subtrack
    return root / task / experiment_name


def to_project_relative_path(path_like: str | Path) -> str:
    path = resolve_project_path(path_like)
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def normalize_path_args(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: to_project_relative_path(value) if key in PATH_ARG_KEYS and value not in (None, "") else value
        for key, value in values.items()
    }


def build_experiment_name(args: argparse.Namespace) -> str:
    feature_tag = f"{args.audio_feature}__{args.video_feature}"
    threshold_tag = "ord5" if args.task == "binary" else "ord5-10"
    encoder_tag = (
        f"depformer_text_coral_d{args.depformer_d_model}"
        f"_bct{args.depformer_bct_layers}"
        f"_h{args.depformer_heads}"
        f"_{threshold_tag}_rw{args.lambda_reg:g}"
    )
    if args.sample_pooling != "mean":
        encoder_tag = f"{encoder_tag}_pool{args.sample_pooling}"
    return args.experiment_name or f"{args.track.lower()}_{args.task}_{args.subtrack}_{encoder_tag}_{feature_tag}"


def append_summary_row(csv_path: Path, row: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in METRIC_ARRAY_KEYS}


def phq_to_binary_ternary(phq9: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    binary = (phq9 >= TASK_THRESHOLDS["binary"][0]).long()
    ternary = (phq9 >= TASK_THRESHOLDS["ternary"][0]).long() + (phq9 >= TASK_THRESHOLDS["ternary"][1]).long()
    return binary, ternary


def coral_reg_loss(
    outputs: dict[str, torch.Tensor],
    phq9: torch.Tensor,
    lambda_reg: float,
    thresholds: tuple[float, ...],
) -> tuple[torch.Tensor, dict[str, float]]:
    phq9 = phq9.float().clamp(0.0, PHQ_MAX)
    ordinal_targets = build_phq_ordinal_targets(phq9, thresholds)
    loss_ord = F.binary_cross_entropy_with_logits(outputs["coral_logits"], ordinal_targets)
    loss_reg = F.smooth_l1_loss(outputs["phq_pred"] / PHQ_MAX, phq9 / PHQ_MAX)
    total = loss_ord + lambda_reg * loss_reg
    return total, {
        "ordinal_loss": float(loss_ord.item()),
        "reg_loss": float(loss_reg.item()),
    }


def forward_batch(model: nn.Module, batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
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
        text_segments=batch["text_segments"].to(device),
        text_mask=batch["text_mask"].to(device),
    )


def configure_epoch_dataset(dataset: Any, epoch: int, training: bool) -> None:
    if hasattr(dataset, "set_chunk_sampling_enabled"):
        dataset.set_chunk_sampling_enabled(training)
    if hasattr(dataset, "set_epoch"):
        dataset.set_epoch(epoch)


def apply_selection_score(metrics: dict[str, Any]) -> dict[str, Any]:
    metrics["competition_score"] = (
        float(metrics.get("f1", 0.0))
        + float(metrics.get("kappa", 0.0))
        + float(metrics.get("ccc", 0.0))
    ) / 3.0
    metrics["selection_score"] = metrics["competition_score"]
    metrics["competition_score"] = metrics["selection_score"]
    return metrics


def evaluate_text_coral_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lambda_reg: float,
    task: str,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_ordinal_loss = 0.0
    total_reg_loss = 0.0
    ids: list[int] = []
    binary_true: list[int] = []
    ternary_true: list[int] = []
    binary_pred: list[int] = []
    ternary_pred: list[int] = []
    phq_true: list[float] = []
    phq_pred: list[float] = []

    thresholds = TASK_THRESHOLDS[task]
    with torch.no_grad():
        for batch in loader:
            outputs = forward_batch(model, batch, device)
            phq9 = batch["phq9"].to(device)
            loss, parts = coral_reg_loss(outputs, phq9, lambda_reg, thresholds)
            batch_size = int(phq9.numel())
            total_loss += float(loss.item()) * batch_size
            total_ordinal_loss += parts["ordinal_loss"] * batch_size
            total_reg_loss += parts["reg_loss"] * batch_size

            true_binary, true_ternary = phq_to_binary_ternary(phq9)
            if task == "binary":
                pred_binary = predict_binary_from_coral(outputs["coral_logits"])
                pred_ternary = pred_binary
            else:
                pred_binary, pred_ternary = predict_binary_ternary_from_coral(outputs["coral_logits"])
            ids.extend(batch["pid"].cpu().numpy().astype(int).tolist())
            binary_true.extend(true_binary.cpu().numpy().astype(int).tolist())
            ternary_true.extend(true_ternary.cpu().numpy().astype(int).tolist())
            binary_pred.extend(pred_binary.cpu().numpy().astype(int).tolist())
            ternary_pred.extend(pred_ternary.cpu().numpy().astype(int).tolist())
            phq_true.extend(phq9.cpu().numpy().astype(float).tolist())
            phq_pred.extend(outputs["phq_pred"].cpu().numpy().astype(float).tolist())

    n_items = max(1, len(ids))
    binary_metrics = classification_metrics(np.asarray(binary_true), np.asarray(binary_pred))
    ternary_metrics = classification_metrics(np.asarray(ternary_true), np.asarray(ternary_pred))
    reg_metrics = regression_metrics(np.asarray(phq_true, dtype=np.float64), np.asarray(phq_pred, dtype=np.float64))
    primary_metrics = binary_metrics if task == "binary" else ternary_metrics
    primary_true = binary_true if task == "binary" else ternary_true
    primary_pred = binary_pred if task == "binary" else ternary_pred
    metrics: dict[str, Any] = {
        "f1": primary_metrics["f1"],
        "acc": primary_metrics["acc"],
        "kappa": primary_metrics["kappa"],
        "binary_f1": binary_metrics["f1"],
        "binary_acc": binary_metrics["acc"],
        "binary_kappa": binary_metrics["kappa"],
        "ternary_f1": ternary_metrics["f1"],
        "ternary_acc": ternary_metrics["acc"],
        "ternary_kappa": ternary_metrics["kappa"],
        "ccc": reg_metrics["ccc"],
        "rmse": reg_metrics["rmse"],
        "mae": reg_metrics["mae"],
        "r2": reg_metrics["r2"],
        "loss": safe_float(total_loss / n_items),
        "ordinal_loss": safe_float(total_ordinal_loss / n_items),
        "reg_loss": safe_float(total_reg_loss / n_items),
        "ids": ids,
        "class_true": primary_true,
        "class_pred": primary_pred,
        "phq_true": phq_true,
        "phq_pred": phq_pred,
        "y_true": primary_true,
        "y_pred": primary_pred,
    }
    return apply_selection_score(metrics)


def main() -> None:
    args = parse_args()
    experiment_name = build_experiment_name(args)
    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    checkpoints_root = resolve_project_path(args.checkpoints_dir)
    logs_root = resolve_project_path(args.logs_dir)
    checkpoints_dir = resolve_track_task_dir(checkpoints_root, args.track, args.subtrack, args.task, experiment_name)
    log_dir = resolve_track_task_dir(logs_root, args.track, args.subtrack, args.task, experiment_name)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(log_dir / f"result_{timestamp}.log")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    thresholds = TASK_THRESHOLDS[args.task]
    split_payload = create_train_val_split(
        split_csv=args.split_csv,
        task=args.task,
        val_ratio=args.val_ratio,
        regression_label="label3",
        seed=args.seed,
        fixed_val_ids_path=args.fixed_val_ids_path,
        fixed_split_label=args.fixed_split_label,
    )
    setup_seed(args.seed)

    train_dataset = MPDDElderTextDataset(
        data_root=args.data_root,
        label_map=split_payload["train_map"],
        source_split_map=split_payload["source_split_map"],
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        text_embedding_npy=args.text_embedding_npy,
        phq_map=split_payload.get("train_phq_map"),
        target_t=args.target_t,
    )
    val_dataset = MPDDElderTextDataset(
        data_root=args.data_root,
        label_map=split_payload["val_map"],
        source_split_map=split_payload["source_split_map"],
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        text_embedding_npy=args.text_embedding_npy,
        phq_map=split_payload.get("val_phq_map"),
        target_t=args.target_t,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )

    input_dims = infer_input_dims(train_dataset)
    current_wav2vec_dim = 768 if args.track == "Track1" else 1024
    model_kwargs = {
        "subtrack": args.subtrack,
        "num_classes": 2 if args.task == "binary" else 3,
        "is_regression": False,
        "use_regression_head": True,
        "mfcc_dim": 64,
        "opensmile_dim": 65,
        "wav2vec_dim": current_wav2vec_dim,
        "densenet_dim": 1000,
        "resnet_dim": 1000,
        "openface_dim": 710,
        "gait_dim": input_dims.get("gait_dim", 12),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "encoder_type": args.encoder_type,
        "text_dim": input_dims.get("text_dim", 1024),
        "depformer_d_model": args.depformer_d_model,
        "depformer_adapter_dim": args.depformer_adapter_dim,
        "depformer_lstm_layers": args.depformer_lstm_layers,
        "depformer_bct_layers": args.depformer_bct_layers,
        "depformer_heads": args.depformer_heads,
        "sample_pooling": args.sample_pooling,
        "coral_threshold_count": len(thresholds),
    }
    model = MPDDDepFormerTextCoral(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    logger.info("Experiment: %s", experiment_name)
    logger.info("Backbone: %s", args.backbone)
    logger.info("Sample pooling: %s", args.sample_pooling)
    logger.info("Device: %s", device)
    logger.info("Train/Val: %d / %d", len(train_dataset), len(val_dataset))
    logger.info("PHQ thresholds: %s | lambda_reg=%.4g", ",".join(f"{v:g}" for v in thresholds), args.lambda_reg)

    history_rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_epoch = 0
    best_val_metrics: dict[str, Any] | None = None
    best_checkpoint_path = checkpoints_dir / f"best_model_{timestamp}.pth"
    epochs_without_improve = 0

    for epoch in range(1, args.epochs + 1):
        configure_epoch_dataset(train_dataset, epoch, training=True)
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = forward_batch(model, batch, device)
            phq9 = batch["phq9"].to(device)
            loss, _parts = coral_reg_loss(outputs, phq9, args.lambda_reg, thresholds)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += float(loss.item()) * int(phq9.numel())

        scheduler.step()
        train_loss = running_loss / max(1, len(train_dataset))
        configure_epoch_dataset(val_dataset, epoch, training=False)
        val_metrics = evaluate_text_coral_model(model, val_loader, device, args.lambda_reg, args.task)
        history_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_competition_score": round(val_metrics["selection_score"], 6),
            "val_binary_f1": round(val_metrics["binary_f1"], 6),
            "val_binary_acc": round(val_metrics["binary_acc"], 6),
            "val_binary_kappa": round(val_metrics["binary_kappa"], 6),
            "val_ternary_f1": round(val_metrics["ternary_f1"], 6),
            "val_ternary_acc": round(val_metrics["ternary_acc"], 6),
            "val_ternary_kappa": round(val_metrics["ternary_kappa"], 6),
            "val_ccc": round(val_metrics["ccc"], 6),
            "val_rmse": round(val_metrics["rmse"], 6),
            "val_mae": round(val_metrics["mae"], 6),
            "val_ordinal_loss": round(val_metrics["ordinal_loss"], 6),
            "val_reg_loss": round(val_metrics["reg_loss"], 6),
        }
        history_rows.append(history_row)
        logger.info(
            "Epoch %d/%d | train_loss=%.6f | bin_f1=%.6f tri_f1=%.6f "
            "ccc=%.6f rmse=%.6f mae=%.6f",
            epoch,
            args.epochs,
            train_loss,
            val_metrics["binary_f1"],
            val_metrics["ternary_f1"],
            val_metrics["ccc"],
            val_metrics["rmse"],
            val_metrics["mae"],
        )

        current_score = float(val_metrics["selection_score"])
        if current_score > best_score + args.min_delta:
            best_score = current_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improve = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_kwargs": model_kwargs,
                    "track": args.track,
                    "task": args.task,
                    "subtrack": args.subtrack,
                    "backbone": args.backbone,
                    "encoder_type": args.encoder_type,
                    "audio_feature": args.audio_feature,
                    "video_feature": args.video_feature,
                    "data_root": to_project_relative_path(args.data_root),
                    "split_csv": to_project_relative_path(args.split_csv),
                    "personality_npy": to_project_relative_path(args.personality_npy),
                    "text_embedding_npy": to_project_relative_path(args.text_embedding_npy),
                    "target_t": args.target_t,
                    "sample_pooling": args.sample_pooling,
                    "seed": args.seed,
                    "lambda_reg": args.lambda_reg,
                    "phq_thresholds": thresholds,
                    "phq_max": PHQ_MAX,
                    "experiment_name": experiment_name,
                    "best_epoch": epoch,
                    "best_val_metrics": summarize_metrics(val_metrics),
                    "metric_split": "val",
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= args.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    if best_val_metrics is None:
        raise RuntimeError("Training finished without a valid validation checkpoint.")
    best_val_summary = summarize_metrics(best_val_metrics)
    history_path = log_dir / f"history_{timestamp}.csv"
    with open(history_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    best_checkpoint_rel = to_project_relative_path(best_checkpoint_path)
    history_rel = to_project_relative_path(history_path)
    result_payload = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "task": args.task,
        "track": args.track,
        "subtrack": args.subtrack,
        "backbone": args.backbone,
        "best_epoch": best_epoch,
        "selection_metric": f"{args.task}_f1_kappa_phq_score",
        "best_val_metrics": best_val_summary,
        "checkpoint_path": best_checkpoint_rel,
        "history_path": history_rel,
        "train_count": len(train_dataset),
        "val_count": len(val_dataset),
        "config": normalize_path_args(vars(args)),
    }
    result_path = log_dir / f"train_result_{timestamp}.json"
    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    summary_row = {
        "timestamp": timestamp,
        "task": args.task,
        "track": args.track,
        "subtrack": args.subtrack,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "checkpoint_path": best_checkpoint_rel,
        "metric_split": "val",
        "selection_metric": f"{args.task}_f1_kappa_phq_score",
        "selection_score": f"{best_val_summary.get('selection_score', 0.0):.6f}",
        "Binary-F1": f"{best_val_summary.get('binary_f1', 0.0):.6f}",
        "Binary-ACC": f"{best_val_summary.get('binary_acc', 0.0):.6f}",
        "Binary-Kappa": f"{best_val_summary.get('binary_kappa', 0.0):.6f}",
        "Ternary-F1": f"{best_val_summary.get('ternary_f1', 0.0):.6f}",
        "Ternary-ACC": f"{best_val_summary.get('ternary_acc', 0.0):.6f}",
        "Ternary-Kappa": f"{best_val_summary.get('ternary_kappa', 0.0):.6f}",
        "CCC": f"{best_val_summary['ccc']:.6f}",
        "RMSE": f"{best_val_summary['rmse']:.6f}",
        "MAE": f"{best_val_summary['mae']:.6f}",
        "R2": f"{best_val_summary.get('r2', 0.0):.6f}",
    }
    append_summary_row(log_dir / f"{experiment_name}.csv", summary_row)
    logger.info("Best checkpoint: %s", best_checkpoint_rel)
    logger.info("Validation metrics saved to: %s", to_project_relative_path(result_path))


if __name__ == "__main__":
    main()
