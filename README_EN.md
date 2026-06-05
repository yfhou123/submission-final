# MPDD-AVG 2026 Elder Track Chunk-Full Experiment Notes

## 1. Directory Structure

The main files and directories are listed below:

```text
submission-final/
├── Elder-trainval/                         # Elder train/validation features, labels, personality features, and text features
│   ├── split_labels_train.csv
│   ├── descriptions.csv
│   ├── descriptions_embeddings_with_ids.npy
│   ├── sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy
│   ├── cached_av_chunks_c4_t128_mfcc2600/
│   └── cached_av_features_t128/
├── Elder-test/                             # Elder test features, text features, and cached features
│   ├── sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy
│   ├── sample_transcriptions_qwen_asr.csv
│   ├── cached_av_chunks_c4_t128_mfcc2600/
│   └── cached_av_features_t128/
├── models/                                 # DepFormer, text-fusion modules, CORAL heads, etc.
├── scripts/
│   ├── cache_elder_av_chunks.py
│   ├── make_elder_cv5_splits.py
│   ├── run_chunkfull_cv5x20_binary.sh
│   ├── run_chunkfull_cv5x20_ternary.sh
│   └── run_chunkfull_fulltrain.sh
├── train_text_coral_attn_chunk_cache.py     # chunk-full training entry
├── train_text_coral_attn_chunk_fulltrain.py # chunk-full full-train entry
├── train_score_loss_text_coral.py           # main CORAL + PHQ regression training logic
├── dataset_text_elder_chunk_cached.py       # Elder dataset that reads chunk cache
├── predict_chunkfull_5vote.py               # predict the test set with 5-model voting
├── select_best5_and_predict.py              # automatically select top-5 checkpoints from training logs and predict
├── qwen_phq9_correction.py                  # Qwen PHQ-9 post-processing correction
├── checkpoint/                              # prepared checkpoints for 5-model voting
├── predict/                                 # 5-model voting output
└── qwen_phq9_corrected/                     # Qwen-corrected output
```

Checkpoint reproduction is described in Section 3.

## 2. Environment Requirements

It is recommended to create a new Python 3.10 environment:

```bash
conda create -n mpddavg python=3.10 -y
conda activate mpddavg
```

Install a PyTorch build matching the local CUDA version first. The following command is for CUDA 12.1. If the machine uses a different CUDA version, replace it with the corresponding PyTorch installation command:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Training and test-set prediction require at least one CUDA GPU. In this experiment, training used an `a100-pcie-40gb`, and Qwen2.5-72B-Instruct-AWQ PHQ-9 label correction used an `nvidia_h100`.

The training scripts use the following defaults:

```bash
PYTHON_BIN=python
DEVICE=cuda
```

To specify a Python executable or device, override them when running the script:

```bash
PYTHON_BIN=/path/to/python DEVICE=cuda bash scripts/run_chunkfull_cv5x20_binary.sh
```

If errors such as `libnvJitLink.so.12` or missing CUDA dynamic libraries occur, first check whether the PyTorch CUDA wheel is compatible with the machine driver. The scripts also try to add common Python CUDA wheel library directories to `LD_LIBRARY_PATH`.

## 3. Predicting the Test Set with Checkpoints and 5-Model Voting

The directories `checkpoint/binary/` and `checkpoint/ternary/` already contain the checkpoints used for test-set prediction. `predict_chunkfull_5vote.py` reads 5 binary checkpoints and 5 ternary checkpoints, runs forward prediction on the test set, and applies majority voting to the classification outputs.

The binary and ternary labels are obtained from their corresponding 5-model votes. The original `phq9_pred` is not voted. Instead, it is taken from the regression head of the following ternary checkpoint:

```text
checkpoint/ternary/04_modelid21_seed17_best_model_2026-06-03-13.31.36.pth
```

The `phq9_pred` produced by this checkpoint is written to both `predict/binary.csv` and `predict/ternary.csv` as the original PHQ-9 prediction before Qwen PHQ-9 correction. The Qwen correction workflow is described in Section 9.

Before prediction, generate the test-set chunk cache according to Section 5 and make sure `Elder-test/cached_av_chunks_c4_t128_mfcc2600/` exists. Otherwise the prediction script cannot read the test-set audio/video input.

Reproduce prediction with:

```bash
cd submission-final

python predict_chunkfull_5vote.py \
  --device cuda
```

For a workflow check, CPU can also be used:

```bash
python predict_chunkfull_5vote.py \
  --device cpu
```

Prediction outputs are saved to:

```text
predict/binary.csv
predict/ternary.csv
```

After completing 5fold * 20seed training, an additional automatic 5-model voting step can be run. It selects models according to validation metrics. This workflow also requires the test-set chunk cache to be generated first:

```bash
cd submission-final

python select_best5_and_predict.py \
  --logs_root logs/cv5x20_chunkfull \
  --output_dir predict_auto_selected \
  --device cuda
```

The script scans:

```text
logs/cv5x20_chunkfull/binary/**/train_result_*.json
logs/cv5x20_chunkfull/ternary/**/train_result_*.json
```

For binary and ternary tasks, the top 5 checkpoints are selected independently by validation `selection_score` and then majority-voted. `phq9_pred` is not voted; instead, the candidate checkpoint with the highest validation `CCC` is selected, and its regression head output is used as the PHQ-9 prediction. Automatically selected outputs are saved to:

```text
predict_auto_selected/binary.csv
predict_auto_selected/ternary.csv
```

## 4. Model Overview

This experiment uses the Elder chunk-full architecture:

1. Each subject has multiple samples.
2. The long audio/video sequence of each sample is pre-split into several chunks.
3. Inside each chunk, six audio/video feature streams are resized to `T=128`:
   - audio: `mfcc`, `opensmile`, `wav2vec`
   - video: `densenet`, `resnet`, `openface`
4. Each chunk is first passed through the DepFormer audio-video interaction module to obtain a chunk-level A/V representation.
5. Attention pooling is applied over chunks from the same sample to obtain a sample-level A/V representation.
6. ASR text is semantically segmented and encoded by a Chinese text encoder into sentence vectors.
7. The personality vector and text sentence vectors interact through cross-attention, producing a sample-level personality-text interaction representation.
8. The sample-level A/V representation and sample-level personality-text interaction representation are concatenated.
9. Attention pooling is applied over multiple samples of the same subject to obtain the subject-level representation.
10. The output heads use a simplified CORAL classification head and a PHQ regression head:
    - binary: one threshold, PHQ >= 5
    - ternary: two thresholds, PHQ >= 5 and PHQ >= 10
    - the regression head outputs a continuous PHQ-9 value

The training loss is:

```text
L = L_ord + lambda_reg * L_reg
```

where:

```text
L_ord = BCE ordinal threshold loss
L_reg = SmoothL1(PHQ_pred / 27, PHQ / 27)
```

The default `lambda_reg` is `5.0`.

### ASR Transcription and Text Embedding Generation

The text modality comes from a locally deployed Qwen3-ASR model. The procedure is: first, Qwen3-ASR is used to transcribe each sample from the privacy-constrained raw audio, producing `sample_transcriptions_qwen_asr.csv`. Then `MPDD/MPDD-AVG-2026/feature_extract/text/segment_asr_transcripts.py` is used to segment each transcript according to Chinese punctuation, minimum length, target length, and maximum segment count, producing `sample_transcription_segments_qwen_asr.csv`. Finally, `MPDD/MPDD-AVG-2026/feature_extract/text/extract_asr_segment_embeddings.py` reads these text segments and uses the local Chinese text encoder `BAAI/bge-large-zh-v1.5` to encode multiple segments from each sample into fixed-limit sentence vectors, masks, and `segment_count`, saved as `sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy`.

The ASR, segmentation, and text encoding steps have already been completed in this submission package, and the corresponding files can be used directly.

## 5. Generate the Chunk-Full A/V Cache

Chunk-full training does not directly read the complete long-sequence features of each sample during training. Instead, audio/video features are preprocessed into a chunk cache. To keep the repository size manageable, cache files are not stored in the repository. Before running the following commands, place the test and training `Video` and `Audio` directories under the corresponding folders:

```text
MPDD/MPDD-AVG-2026/submission-final/Elder-test/Audio
MPDD/MPDD-AVG-2026/submission-final/Elder-test/Video
MPDD/MPDD-AVG-2026/submission-final/Elder-trainval/Audio
MPDD/MPDD-AVG-2026/submission-final/Elder-trainval/Video
```

```bash
cd submission-final

# Train-set cache: use the training label CSV to provide IDs and split information.
python scripts/cache_elder_av_chunks.py \
  --data_root Elder-trainval \
  --split_csv Elder-trainval/split_labels_train.csv \
  --personality_npy Elder-trainval/descriptions_embeddings_with_ids.npy \
  --output_dir Elder-trainval/cached_av_chunks_c4_t128_mfcc2600 \
  --target_t 128 \
  --max_chunks 4 \
  --overwrite

# Test-set cache: no label CSV is needed; IDs are inferred from Elder-test/Audio and Elder-test/Video.
python scripts/cache_elder_av_chunks.py \
  --data_root Elder-test \
  --split_csv '' \
  --split_name test \
  --personality_npy Elder-trainval/descriptions_embeddings_with_ids.npy \
  --output_dir Elder-test/cached_av_chunks_c4_t128_mfcc2600 \
  --target_t 128 \
  --max_chunks 4 \
  --overwrite
```

The main logic of `scripts/cache_elder_av_chunks.py` is:

1. If `--split_csv` is provided, read the CSV to obtain subject IDs, labels, and split information. If `--split_csv ''` is used, infer subject IDs from the ID subdirectories under `Audio/mfcc` and `Video/densenet`.
2. Use `MPDDElderDataset` to locate the audio and video feature files for multiple samples under each subject.
3. For each sample, first read the MFCC frame count and determine the number of chunks according to MFCC length:

```text
mfcc_frames <= 2600  -> 1 chunk
mfcc_frames <= 5200  -> 2 chunks
mfcc_frames <= 7800  -> 3 chunks
mfcc_frames >  7800  -> 4 chunks
```

4. The six audio/video feature streams share the chunk boundaries determined by MFCC:

```text
mfcc, opensmile, wav2vec, densenet, resnet, openface
```

5. Each chunk is resized internally to `target_t=128`.
6. If a sample has fewer than `max_chunks=4` chunks, zero padding is applied to 4 chunks.
7. If a subject has fewer than `PAIR_COUNT=4` samples, zero padding is also applied to 4 samples.
8. Each subject is saved as one `.pt` file containing:

```text
mfcc / opensmile / wav2vec / densenet / resnet / openface: [P, C, T, D]
pair_mask: [P]
chunk_mask: [P, C]
chunk_count: [P]
```

Here, `P=4` means at most 4 samples, `C=4` means at most 4 chunks, and `T=128` means the time length after resizing each chunk. During training, `dataset_text_elder_chunk_cached.py` directly reads these `.pt` files, so the long sequences do not need to be split and resized repeatedly during training.

## 6. Generate 5-Fold Splits

The training scripts automatically call `scripts/make_elder_cv5_splits.py` to generate 5 folds. To generate them manually:

```bash
cd submission-final

python scripts/make_elder_cv5_splits.py \
  --split_csv Elder-trainval/split_labels_train.csv \
  --out_dir splits/elder_cv5_label3_seed3407 \
  --label_col label3 \
  --seed 3407 \
  --overwrite
```

By default, stratification is based on the ternary `label3`, and binary and ternary experiments share the same folds.

## 7. Run 5fold * 20seed Training

The scripts use the following 20 random seeds:

```text
3407 42 2026 17 29 73 101 233 521 777
1009 1234 1601 1997 2701 4099 5051 6067 7411 9001
```

Binary training:

```bash
cd submission-final

bash scripts/run_chunkfull_cv5x20_binary.sh
```

Ternary training:

```bash
cd submission-final

bash scripts/run_chunkfull_cv5x20_ternary.sh
```

Common overridable parameters:

```bash
EPOCHS=80
BATCH_SIZE=4
LR=5e-4
PATIENCE=25
LAMBDA_REG=5.0
DEVICE=cuda
```

Example:

```bash
EPOCHS=100 PATIENCE=30 BATCH_SIZE=4 bash scripts/run_chunkfull_cv5x20_ternary.sh
```

In this experiment, one `a100-pcie-40gb` GPU was used for 5folds * 20seeds training with `EPOCHS=100` and `PATIENCE=30`.

## 8. Checkpoint and Log Paths

Binary checkpoints:

```text
checkpoints/cv5x20_chunkfull/binary/elder_chunkfull_c4t128_cv5x20_fold{fold}_seed{seed}/
```

Ternary checkpoints:

```text
checkpoints/cv5x20_chunkfull/ternary/elder_chunkfull_c4t128_cv5x20_fold{fold}_seed{seed}/
```

Log paths:

```text
logs/cv5x20_chunkfull/binary/elder_chunkfull_c4t128_cv5x20_fold{fold}_seed{seed}/
logs/cv5x20_chunkfull/ternary/elder_chunkfull_c4t128_cv5x20_fold{fold}_seed{seed}/
```

Each fold/seed has an independent directory, which makes later statistics and model selection easier.

### Full-Training with a Fixed Epoch

Additional experiments were also conducted. After the regular train/validation split experiments, the Elder dataset size was considered relatively small. Splitting a validation set from the training data further reduces the amount of data available for parameter updates and can introduce more instability from the split. Therefore, a full-training setup was also tested, where all training samples were used for parameter updates and several fixed epoch snapshots were evaluated. Candidate epochs included `10, 13, 17, 18, 24, 26, 37, 48, 52, 59`. In these experiments, strong single-model binary settings included `seed=1009, epoch=17` and `seed=777, epoch=18`; strong single-model ternary settings included `seed=2701, epoch=37` and `seed=3583, epoch=52`.

To train on the full training set without splitting out a validation set, use `scripts/run_chunkfull_fulltrain.sh`.

Binary example:

```bash
cd submission-final

TASK=binary TRAIN_SEEDS='1009' EPOCHS=17 bash scripts/run_chunkfull_fulltrain.sh
```

Ternary example:

```bash
cd submission-final

TASK=ternary TRAIN_SEEDS='2701' EPOCHS=37 bash scripts/run_chunkfull_fulltrain.sh
```

## 9. Qwen PHQ-9 Post-Processing Correction

`qwen_phq9_correction.py` reads:

```text
predict/binary.csv
predict/ternary.csv
Elder-trainval/descriptions.csv
Elder-test/sample_transcriptions_qwen_asr.csv
```

and calls the local model:

```text
models/Qwen2.5-72B-Instruct-AWQ
```

This experiment used one `nvidia_h100` GPU to correct the PHQ-9 values output by the model.

If the model is not available locally, download it from the `submission-final` root directory:

```bash
cd submission-final

hf download Qwen/Qwen2.5-72B-Instruct-AWQ \
  --local-dir models/Qwen2.5-72B-Instruct-AWQ
```

If HuggingFace access is slow in the current network environment, configure a mirror first and then run the same download command:

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-72B-Instruct-AWQ \
  --local-dir models/Qwen2.5-72B-Instruct-AWQ
```

For each subject, 10 PHQ-9 predictions are generated. The `phq9` value is parsed from raw text, then the final corrected value is computed as:

```text
corrected_phq9 = 0.9 * mean(qwen_parsed_phq9) + 0.1 * original_phq9_pred
```

Dry-run the prompt first:

```bash
cd submission-final

python qwen_phq9_correction.py \
  --dry_run \
  --limit 1 \
  --local_files_only
```

Formal run:

```bash
CUDA_VISIBLE_DEVICES=0 python \
  qwen_phq9_correction.py \
  --local_files_only \
  --n_samples 10 \
  --temperature 0.3 \
  --top_p 0.9
```
