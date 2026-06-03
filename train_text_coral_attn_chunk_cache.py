from __future__ import annotations

import os
import sys
from typing import Any

import train_score_loss_text_coral as trainer
from dataset_text_elder_chunk_cached import MPDDElderTextChunkCachedDataset, infer_input_dims
from models.depformer_text_chunk_coral_mpdd import MPDDDepFormerTextChunkCoral


class ConfiguredChunkCachedTextDataset(MPDDElderTextChunkCachedDataset):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        av_cache_dir = os.environ.get("AV_CHUNK_CACHE_DIR", "").strip() or None
        if av_cache_dir is None:
            av_cache_dir = os.environ.get("AV_CACHE_DIR", "").strip() or None
        super().__init__(*args, av_cache_dir=av_cache_dir, **kwargs)


def forward_batch_with_chunk_mask(model, batch, device):
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


_original_build_experiment_name = trainer.build_experiment_name


def build_chunk_cached_experiment_name(args: Any) -> str:
    if getattr(args, "experiment_name", ""):
        return args.experiment_name
    return f"{_original_build_experiment_name(args)}_chunkattn_c4t128_cachedav"


def ensure_attention_chunk_defaults(argv: list[str]) -> list[str]:
    additions: list[str] = []
    option_defaults = {
        "--sample_pooling": "attention",
        "--target_t": "128",
    }
    existing = set()
    for arg in argv[1:]:
        if arg.startswith("--"):
            existing.add(arg.split("=", 1)[0])
    for option, value in option_defaults.items():
        if option not in existing:
            additions.extend([option, value])
    return [argv[0], *additions, *argv[1:]]


trainer.MPDDElderTextDataset = ConfiguredChunkCachedTextDataset
trainer.infer_input_dims = infer_input_dims
trainer.MPDDDepFormerTextCoral = MPDDDepFormerTextChunkCoral
trainer.forward_batch = forward_batch_with_chunk_mask
trainer.build_experiment_name = build_chunk_cached_experiment_name


if __name__ == "__main__":
    sys.argv = ensure_attention_chunk_defaults(sys.argv)
    trainer.main()
