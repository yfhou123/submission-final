#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POOL_SPLITS = {"", "train", "val"}


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Elder stratified 5-fold val-id JSON files.")
    parser.add_argument("--split_csv", default="Elder-trainval/split_labels_train.csv")
    parser.add_argument("--out_dir", default="splits/elder_cv5_label3_seed3407")
    parser.add_argument("--label_col", default="label3", choices=["label2", "label3"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_csv = resolve_path(args.split_csv)
    out_dir = resolve_path(args.out_dir)
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("split", "train").strip().lower() in POOL_SPLITS
        ]
    if not rows:
        raise RuntimeError(f"No train rows found in {split_csv}")

    ids = [int(row["ID"]) for row in rows]
    labels = [int(float(row[args.label_col])) for row in rows]
    label_counts = Counter(labels)
    if min(label_counts.values()) < args.n_splits:
        raise RuntimeError(
            f"Cannot create {args.n_splits}-fold stratified split; "
            f"{args.label_col} counts={dict(label_counts)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    all_payloads: list[dict[str, object]] = []
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(ids, labels), start=1):
        train_ids = sorted(ids[index] for index in train_idx)
        val_ids = sorted(ids[index] for index in val_idx)
        path = out_dir / f"fold{fold_idx}.json"
        if path.exists() and not args.overwrite:
            print(f"exists: {path.relative_to(PROJECT_ROOT)}")
            continue
        payload = {
            "split_csv": str(split_csv.relative_to(PROJECT_ROOT)),
            "seed": args.seed,
            "n_splits": args.n_splits,
            "fold": fold_idx,
            "fixed_split_label": args.label_col,
            "train_count": len(train_ids),
            "val_count": len(val_ids),
            "train_ids": train_ids,
            "val_ids": val_ids,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        all_payloads.append(payload)
        print(f"wrote: {path.relative_to(PROJECT_ROOT)} val_count={len(val_ids)}")

    summary_path = out_dir / "summary.json"
    summary = {
        "split_csv": str(split_csv.relative_to(PROJECT_ROOT)),
        "out_dir": str(out_dir.relative_to(PROJECT_ROOT)),
        "seed": args.seed,
        "n_splits": args.n_splits,
        "label_col": args.label_col,
        "label_counts": dict(sorted(label_counts.items())),
        "folds": all_payloads,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary: {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
