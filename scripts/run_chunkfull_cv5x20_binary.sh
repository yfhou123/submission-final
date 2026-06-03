#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-/home/yfhou/miniconda3/envs/mpddavg/bin/python}"
DEVICE="${DEVICE:-cuda}"
TASK="binary"
TRAIN_SEEDS="${TRAIN_SEEDS:-3407 42 2026 17 29 73 101 233 521 777 1009 1234 1601 1997 2701 4099 5051 6067 7411 9001}"
FOLDS="${FOLDS:-1 2 3 4 5}"
SPLIT_SEED="${SPLIT_SEED:-3407}"
SPLIT_ROOT="${SPLIT_ROOT:-splits/elder_cv5_label3_seed${SPLIT_SEED}}"
FIXED_SPLIT_LABEL="${FIXED_SPLIT_LABEL:-label3}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
HIDDEN_DIM="${HIDDEN_DIM:-64}"
DROPOUT="${DROPOUT:-0.4}"
PATIENCE="${PATIENCE:-25}"
MIN_DELTA="${MIN_DELTA:-1e-4}"
TARGET_T="${TARGET_T:-128}"
LAMBDA_REG="${LAMBDA_REG:-5.0}"
SAMPLE_POOLING="${SAMPLE_POOLING:-attention}"

DEPFORMER_D_MODEL="${DEPFORMER_D_MODEL:-256}"
DEPFORMER_ADAPTER_DIM="${DEPFORMER_ADAPTER_DIM:-128}"
DEPFORMER_LSTM_LAYERS="${DEPFORMER_LSTM_LAYERS:-1}"
DEPFORMER_BCT_LAYERS="${DEPFORMER_BCT_LAYERS:-1}"
DEPFORMER_HEADS="${DEPFORMER_HEADS:-2}"

DATA_ROOT="${DATA_ROOT:-Elder-trainval}"
SPLIT_CSV="${SPLIT_CSV:-Elder-trainval/split_labels_train.csv}"
PERSONALITY_NPY="${PERSONALITY_NPY:-Elder-trainval/descriptions_embeddings_with_ids.npy}"
TEXT_EMBEDDING_NPY="${TEXT_EMBEDDING_NPY:-Elder-trainval/sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy}"
AV_CHUNK_CACHE_DIR="${AV_CHUNK_CACHE_DIR:-Elder-trainval/cached_av_chunks_c4_t128_mfcc2600}"
CHECKPOINTS_DIR="${CHECKPOINTS_DIR:-checkpoints/cv5x20_chunkfull}"
LOGS_DIR="${LOGS_DIR:-logs/cv5x20_chunkfull}"
EXP_NAME="${EXP_NAME:-elder_chunkfull_c4t128_cv5x20}"

NVIDIA_LIB_DIRS=(
  "$HOME/.local/lib/python3.10/site-packages/nvidia/nvjitlink/lib"
  "$HOME/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cublas/lib"
  "$HOME/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cudnn/lib"
  "$HOME/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cuda_runtime/lib"
  "$HOME/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cusparse/lib"
  "$HOME/miniconda3/envs/mpddavg/lib/python3.10/site-packages/nvidia/cusolver/lib"
)
for lib_dir in "${NVIDIA_LIB_DIRS[@]}"; do
  if [[ -d "$lib_dir" ]]; then
    export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
  fi
done

cd "$PROJECT_ROOT"

for required_path in "$DATA_ROOT" "$SPLIT_CSV" "$PERSONALITY_NPY" "$TEXT_EMBEDDING_NPY" "$AV_CHUNK_CACHE_DIR"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required path: $required_path" >&2
    exit 1
  fi
done

"$PYTHON_BIN" scripts/make_elder_cv5_splits.py \
  --split_csv "$SPLIT_CSV" \
  --out_dir "$SPLIT_ROOT" \
  --label_col "$FIXED_SPLIT_LABEL" \
  --seed "$SPLIT_SEED"

export AV_CHUNK_CACHE_DIR
export AV_CACHE_DIR="$AV_CHUNK_CACHE_DIR"

EXTRA_ARGS=("$@")

for fold in $FOLDS; do
  fixed_val_ids="${SPLIT_ROOT}/fold${fold}.json"
  if [[ ! -f "$fixed_val_ids" ]]; then
    echo "Missing fold split: $fixed_val_ids" >&2
    exit 1
  fi
  for run_seed in $TRAIN_SEEDS; do
    run_exp_name="${EXP_NAME}_fold${fold}_seed${run_seed}"
    echo "[chunkfull-cv5x20][$TASK] fold=$fold seed=$run_seed split=$fixed_val_ids"
    "$PYTHON_BIN" train_text_coral_attn_chunk_cache.py \
      --config config.json \
      --track Track1 \
      --task "$TASK" \
      --subtrack A-V+P \
      --encoder_type bilstm_mean \
      --backbone depformer_text_coral \
      --audio_feature all_audio \
      --video_feature all_video \
      --data_root "$DATA_ROOT" \
      --split_csv "$SPLIT_CSV" \
      --personality_npy "$PERSONALITY_NPY" \
      --text_embedding_npy "$TEXT_EMBEDDING_NPY" \
      --depformer_d_model "$DEPFORMER_D_MODEL" \
      --depformer_adapter_dim "$DEPFORMER_ADAPTER_DIM" \
      --depformer_lstm_layers "$DEPFORMER_LSTM_LAYERS" \
      --depformer_bct_layers "$DEPFORMER_BCT_LAYERS" \
      --depformer_heads "$DEPFORMER_HEADS" \
      --sample_pooling "$SAMPLE_POOLING" \
      --lambda_reg "$LAMBDA_REG" \
      --seed "$run_seed" \
      --epochs "$EPOCHS" \
      --batch_size "$BATCH_SIZE" \
      --lr "$LR" \
      --weight_decay "$WEIGHT_DECAY" \
      --hidden_dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --patience "$PATIENCE" \
      --min_delta "$MIN_DELTA" \
      --target_t "$TARGET_T" \
      --fixed_val_ids_path "$fixed_val_ids" \
      --fixed_split_label "$FIXED_SPLIT_LABEL" \
      --checkpoints_dir "$CHECKPOINTS_DIR" \
      --logs_dir "$LOGS_DIR" \
      --experiment_name "$run_exp_name" \
      --device "$DEVICE" \
      "${EXTRA_ARGS[@]}"
  done
done
