from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import train_text_coral_attn_chunk_cache as chunk_entry
from dataset import get_phq9_target, get_task_label
from train_val_split import _load_train_rows

import torch
from torch import nn
from torch.utils.data import DataLoader


trainer = chunk_entry.trainer


def build_full_train_maps(split_csv: str | Path, task: str) -> dict[str, Any]:
    rows = _load_train_rows(split_csv)
    label_map = {int(row["ID"]): get_task_label(row, task, "label3") for row in rows}
    phq_map = {int(row["ID"]): get_phq9_target(row) for row in rows}
    source_split_map = {int(row["ID"]): "train" for row in rows}
    return {
        "rows": rows,
        "label_map": label_map,
        "phq_map": phq_map,
        "source_split_map": source_split_map,
    }


def main() -> None:
    args = trainer.parse_args()
    experiment_name = trainer.build_experiment_name(args)
    timestamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    checkpoints_root = trainer.resolve_project_path(args.checkpoints_dir)
    logs_root = trainer.resolve_project_path(args.logs_dir)
    checkpoints_dir = trainer.resolve_track_task_dir(
        checkpoints_root,
        args.track,
        args.subtrack,
        args.task,
        experiment_name,
    )
    log_dir = trainer.resolve_track_task_dir(logs_root, args.track, args.subtrack, args.task, experiment_name)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = trainer.setup_logger(log_dir / f"result_{timestamp}.log")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    thresholds = trainer.TASK_THRESHOLDS[args.task]
    full_payload = build_full_train_maps(args.split_csv, args.task)
    trainer.setup_seed(args.seed)

    train_dataset = trainer.MPDDElderTextDataset(
        data_root=args.data_root,
        label_map=full_payload["label_map"],
        source_split_map=full_payload["source_split_map"],
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=args.audio_feature,
        video_feature=args.video_feature,
        personality_npy=args.personality_npy,
        text_embedding_npy=args.text_embedding_npy,
        phq_map=full_payload["phq_map"],
        target_t=args.target_t,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=trainer.collate_batch,
        num_workers=args.num_workers,
    )

    input_dims = trainer.infer_input_dims(train_dataset)
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
    model = trainer.MPDDDepFormerTextCoral(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    logger.info("Experiment: %s", experiment_name)
    logger.info("Backbone: %s", args.backbone)
    logger.info("Sample pooling: %s", args.sample_pooling)
    logger.info("Device: %s", device)
    logger.info("Train mode: full train set, no validation split")
    logger.info("Train count: %d", len(train_dataset))
    logger.info("PHQ thresholds: %s | lambda_reg=%.4g", ",".join(f"{v:g}" for v in thresholds), args.lambda_reg)
    logger.info("Final checkpoint epoch: %d", args.epochs)

    history_rows: list[dict[str, Any]] = []
    final_train_loss = 0.0
    final_ordinal_loss = 0.0
    final_reg_loss = 0.0
    checkpoint_path = checkpoints_dir / f"best_model_{timestamp}.pth"

    for epoch in range(1, args.epochs + 1):
        trainer.configure_epoch_dataset(train_dataset, epoch, training=True)
        model.train()
        running_loss = 0.0
        running_ordinal_loss = 0.0
        running_reg_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            outputs = trainer.forward_batch(model, batch, device)
            phq9 = batch["phq9"].to(device)
            loss, parts = trainer.coral_reg_loss(outputs, phq9, args.lambda_reg, thresholds)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_size = int(phq9.numel())
            running_loss += float(loss.item()) * batch_size
            running_ordinal_loss += parts["ordinal_loss"] * batch_size
            running_reg_loss += parts["reg_loss"] * batch_size

        scheduler.step()
        item_count = max(1, len(train_dataset))
        final_train_loss = running_loss / item_count
        final_ordinal_loss = running_ordinal_loss / item_count
        final_reg_loss = running_reg_loss / item_count
        history_row = {
            "epoch": epoch,
            "train_loss": round(final_train_loss, 6),
            "train_ordinal_loss": round(final_ordinal_loss, 6),
            "train_reg_loss": round(final_reg_loss, 6),
        }
        history_rows.append(history_row)
        logger.info(
            "Epoch %d/%d | train_loss=%.6f | ord=%.6f | reg=%.6f",
            epoch,
            args.epochs,
            final_train_loss,
            final_ordinal_loss,
            final_reg_loss,
        )

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
            "data_root": trainer.to_project_relative_path(args.data_root),
            "split_csv": trainer.to_project_relative_path(args.split_csv),
            "personality_npy": trainer.to_project_relative_path(args.personality_npy),
            "text_embedding_npy": trainer.to_project_relative_path(args.text_embedding_npy),
            "target_t": args.target_t,
            "sample_pooling": args.sample_pooling,
            "seed": args.seed,
            "lambda_reg": args.lambda_reg,
            "phq_thresholds": thresholds,
            "phq_max": trainer.PHQ_MAX,
            "experiment_name": experiment_name,
            "best_epoch": args.epochs,
            "metric_split": "full_train_final_epoch",
        },
        checkpoint_path,
    )

    history_path = log_dir / f"history_{timestamp}.csv"
    with open(history_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    checkpoint_rel = trainer.to_project_relative_path(checkpoint_path)
    history_rel = trainer.to_project_relative_path(history_path)
    result_payload = {
        "experiment_name": experiment_name,
        "timestamp": timestamp,
        "task": args.task,
        "track": args.track,
        "subtrack": args.subtrack,
        "backbone": args.backbone,
        "best_epoch": args.epochs,
        "selection_metric": "fixed_final_epoch",
        "metric_split": "full_train_final_epoch",
        "checkpoint_path": checkpoint_rel,
        "history_path": history_rel,
        "train_count": len(train_dataset),
        "val_count": 0,
        "final_train_loss": final_train_loss,
        "final_train_ordinal_loss": final_ordinal_loss,
        "final_train_reg_loss": final_reg_loss,
        "config": trainer.normalize_path_args(vars(args)),
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
        "best_epoch": args.epochs,
        "checkpoint_path": checkpoint_rel,
        "metric_split": "full_train_final_epoch",
        "selection_metric": "fixed_final_epoch",
        "train_loss": f"{final_train_loss:.6f}",
        "train_ordinal_loss": f"{final_ordinal_loss:.6f}",
        "train_reg_loss": f"{final_reg_loss:.6f}",
    }
    trainer.append_summary_row(log_dir / f"{experiment_name}.csv", summary_row)
    logger.info("Final checkpoint: %s", checkpoint_rel)
    logger.info("Training history saved to: %s", history_rel)
    logger.info("Training result saved to: %s", trainer.to_project_relative_path(result_path))


if __name__ == "__main__":
    main()
