# MPDD-AVG 2026 Elder Track Chunk-Full 实验说明

## 1. 目录结构

主要文件和目录如下：

```text
submission-final/
├── Elder-trainval/                         # Elder 训练集特征、标签、人格和文本特征
│   ├── split_labels_train.csv
│   ├── descriptions.csv
│   ├── descriptions_embeddings_with_ids.npy
│   ├── sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy
│   ├── cached_av_chunks_c4_t128_mfcc2600/
│   └── cached_av_features_t128/
├── Elder-test/                             # Elder 测试集特征、文本特征和缓存特征
│   ├── sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy
│   ├── sample_transcriptions_qwen_asr.csv
│   ├── cached_av_chunks_c4_t128_mfcc2600/
│   └── cached_av_features_t128/
├── models/                                 # DepFormer、文本融合、CORAL head 等模型代码
├── scripts/
│   ├── cache_elder_av_chunks.py
│   ├── make_elder_cv5_splits.py
│   ├── run_chunkfull_cv5x20_binary.sh
│   ├── run_chunkfull_cv5x20_ternary.sh
│   └── run_chunkfull_fulltrain.sh
├── train_text_coral_attn_chunk_cache.py     # chunk-full 训练入口
├── train_text_coral_attn_chunk_fulltrain.py # chunk-full 全量训练入口
├── train_score_loss_text_coral.py           # CORAL + PHQ 回归训练主逻辑
├── dataset_text_elder_chunk_cached.py       # 读取 chunk cache 的 Elder dataset
├── predict_chunkfull_5vote.py               # 5 模型投票预测测试集
├── select_best5_and_predict.py              # 从训练日志自动选 top5 checkpoint 并预测
├── qwen_phq9_correction.py                  # Qwen PHQ-9 后处理校正
├── checkpoint/                              # 已准备的 5 投票 checkpoint
├── predict/                                 # 5 投票输出
└── qwen_phq9_corrected/                     # Qwen 校正后的输出
```

checkpoint 复现相关过程见第 3 节。

## 2. 环境要求

建议新建一个 Python 3.10 环境运行本代码：

```bash
conda create -n mpddavg python=3.10 -y
conda activate mpddavg
```

先安装和本机 CUDA 匹配的 PyTorch。以下命令适用于 CUDA 12.1；如果机器 CUDA 版本不同，请按 PyTorch 官方安装命令替换：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

再安装本项目其余依赖：

```bash
pip install -r requirements.txt
```

训练和测试集预测至少需要一张 CUDA GPU。本试验的训练使用的是a100-pcie-40gb，Qwen2.5-72B-Instruct-AWQ 的 PHQ-9 标签校正使用nvidia_h100 。

训练脚本默认使用：

```bash
PYTHON_BIN=python
DEVICE=cuda
```

如果需要指定 Python 或设备，可以在运行脚本时覆盖：

```bash
PYTHON_BIN=/path/to/python DEVICE=cuda bash scripts/run_chunkfull_cv5x20_binary.sh
```

如果运行时出现 `libnvJitLink.so.12` 或 CUDA 动态库找不到的问题，优先确认 PyTorch CUDA wheel 与机器驱动兼容；脚本中也会自动尝试把常见的 Python CUDA wheel 库目录加入 `LD_LIBRARY_PATH`。

## 3. 使用 checkpoint 预测测试集并进行 5 模型投票

当前目录下 `checkpoint/binary/` 和 `checkpoint/ternary/` 已放置用于测试集预测的 checkpoint。`predict_chunkfull_5vote.py` 会分别读取二分类和三分类各 5 个 checkpoint，对测试集进行前向预测，并对分类结果做 5 模型多数投票。

其中，二分类/三分类标签由各自 5 个模型投票得到；`phq9_pred` 的原始预测值不做投票，而是统一使用下面这个 ternary checkpoint 的回归头输出：

```text
checkpoint/ternary/04_modelid21_seed17_best_model_2026-06-03-13.31.36.pth
```

该 checkpoint 产生的 `phq9_pred` 会同时写入 `predict/binary.csv` 和 `predict/ternary.csv`，作为后续 Qwen PHQ-9 校正前的原始 PHQ-9 预测值。Qwen 校正流程见第 9 节。

预测前需要先按第 5 节生成测试集 chunk cache，即确保 `Elder-test/cached_av_chunks_c4_t128_mfcc2600/` 已存在；否则预测脚本无法读取测试集音视频输入。

复现预测命令：

```bash
cd submission-final

python predict_chunkfull_5vote.py \
  --device cuda
```

如果只想检查流程，也可以使用 CPU：

```bash
python predict_chunkfull_5vote.py \
  --device cpu
```

预测输出保存在：

```text
predict/binary.csv
predict/ternary.csv
```

在完成 5fold * 20seed 训练后，也可以进行一步五模型投票，根据验证集指标自动选择模型。该流程同样要求先生成测试集 chunk cache：

```bash
cd submission-final

python select_best5_and_predict.py \
  --logs_root logs/cv5x20_chunkfull \
  --output_dir predict_auto_selected \
  --device cuda
```

该脚本会分别扫描：

```text
logs/cv5x20_chunkfull/binary/**/train_result_*.json
logs/cv5x20_chunkfull/ternary/**/train_result_*.json
```

二分类和三分类各自按验证集 `selection_score` 选择前 5 个 checkpoint 做多数投票；`phq9_pred` 不做投票，而是在候选模型中选择验证集 `CCC` 最高的单个 checkpoint，用它的回归头输出 PHQ-9。自动选择结果保存在：

```text
predict_auto_selected/binary.csv
predict_auto_selected/ternary.csv
```


## 4. 模型整体思路

本实验使用 Elder 的 chunk-full 架构：

1. 每个 subject 有多个 sample。
2. 每个 sample 的音视频长序列预先切成多个 chunk。
3. 每个 chunk 内部的六种音视频特征被 resize 到 `T=128`：
   - audio: `mfcc`, `opensmile`, `wav2vec`
   - video: `densenet`, `resnet`, `openface`
4. 每个 chunk 先过 DepFormer 音视频交互模块，得到 chunk 级 AV 表示。
5. 对同一个 sample 内的多个 chunk 做 attention pooling，得到 sample 级 AV 表示。
6. ASR 文本已按语义切段并用中文 text encoder 编成句向量。
7. 人格向量与文本句向量做 cross-attention，得到 sample 级人格-文本交互表示。
8. 将 sample 级 AV 表示与 sample 级人格-文本交互表示拼接。
9. 对 subject 内多个 sample 做 attention pooling，得到 subject 级表示。
10. 输出头使用简化 CORAL 分类头和 PHQ 回归头：
    - binary: 一个阈值，PHQ >= 5
    - ternary: 两个阈值，PHQ >= 5 和 PHQ >= 10
    - 回归头输出 PHQ-9 连续值

训练 loss 为：

```text
L = L_ord + lambda_reg * L_reg
```

其中：

```text
L_ord = BCE ordinal threshold loss
L_reg = SmoothL1(PHQ_pred / 27, PHQ / 27)
```

默认 `lambda_reg=5.0`。

### ASR 转录和文本 embedding 生成

文本模态来自本地部署的 Qwen3-ASR。具体做法是先使用 Qwen3-ASR 对脱敏后的原始音频进行逐 sample 转录，得到 `sample_transcriptions_qwen_asr.csv`。随后使用原工程中的 `MPDD/MPDD-AVG-2026/feature_extract/text/segment_asr_transcripts.py` 对每条转录文本按中文标点、最小长度、目标长度和最大 segment 数进行语义切段，生成 `sample_transcription_segments_qwen_asr.csv`。最后使用 `MPDD/MPDD-AVG-2026/feature_extract/text/extract_asr_segment_embeddings.py` 读取这些文本 segment，并调用本地中文文本编码器 `BAAI/bge-large-zh-v1.5` 将每个 sample 的多个 segment 编码成固定上限的句向量、mask 和 segment_count，保存为 `sample_text_segment_embeddings_qwen_asr_bge_large_zh.npy`。

本提交包中已经完成上述 ASR、切段和文本编码步骤，可以直接使用相关文件

## 5. 生成 chunk-full A/V cache

chunk-full 训练不在训练过程中直接读取每个 sample 的完整长序列特征，而是先把音视频特征预处理成 chunk cache。考虑文件大小问题，故没有在仓库中直接存放cache文件，需要用以下指令生成，并在运行前，将测试集和训练集的Video和Audio分别放在对应的文件夹下：

MPDD/MPDD-AVG-2026/submission-final/Elder-test/Audio
MPDD/MPDD-AVG-2026/submission-final/Elder-test/Video
MPDD/MPDD-AVG-2026/submission-final/Elder-trainval/Audio
MPDD/MPDD-AVG-2026/submission-final/Elder-trainval/Video
```bash
cd submission-final

# 训练集 cache：使用训练标签 CSV 提供 ID 和 split 信息。
python scripts/cache_elder_av_chunks.py \
  --data_root Elder-trainval \
  --split_csv Elder-trainval/split_labels_train.csv \
  --personality_npy Elder-trainval/descriptions_embeddings_with_ids.npy \
  --output_dir Elder-trainval/cached_av_chunks_c4_t128_mfcc2600 \
  --target_t 128 \
  --max_chunks 4 \
  --overwrite

# 测试集 cache：不需要标签 CSV，脚本会从 Elder-test/Audio 和 Elder-test/Video 自动推断 ID。
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

`scripts/cache_elder_av_chunks.py` 的主要处理逻辑如下：

1. 如果提供 `--split_csv`，读取 CSV 得到 subject ID、标签和 split 信息；如果 `--split_csv ''`，则从 `Audio/mfcc` 和 `Video/densenet` 的 ID 子目录自动推断 subject ID。
2. 通过 `MPDDElderDataset` 定位每个 subject 下多个 sample 的音频和视频特征文件。
3. 对每个 sample，先读取 MFCC 帧数，并用 MFCC 长度决定该 sample 切成几个 chunk：

```text
mfcc_frames <= 2600  -> 1 chunk
mfcc_frames <= 5200  -> 2 chunks
mfcc_frames <= 7800  -> 3 chunks
mfcc_frames >  7800  -> 4 chunks
```

4. 六种音视频特征共用 MFCC 决定出的 chunk 边界：

```text
mfcc, opensmile, wav2vec, densenet, resnet, openface
```

5. 每个 chunk 内部 resize 到 `target_t=128`。
6. 如果一个 sample 的 chunk 数少于 `max_chunks=4`，则补零到 4 个 chunk。
7. 如果一个 subject 的 sample 数少于 `PAIR_COUNT=4`，也补零到 4 个 sample。
8. 每个 subject 最终保存为一个 `.pt` 文件，文件中包含：

```text
mfcc / opensmile / wav2vec / densenet / resnet / openface: [P, C, T, D]
pair_mask: [P]
chunk_mask: [P, C]
chunk_count: [P]
```

其中 `P=4` 表示最多 4 个 sample，`C=4` 表示最多 4 个 chunk，`T=128` 表示每个 chunk resize 后的时间长度。训练时 `dataset_text_elder_chunk_cached.py` 会直接读取这些 `.pt` 文件，因此训练阶段不需要重复切分和 resize 长序列。

## 6. 生成 5fold 划分

训练脚本会自动调用 `scripts/make_elder_cv5_splits.py` 生成 5fold。如果需要手动生成：

```bash
cd submission-final

python scripts/make_elder_cv5_splits.py \
  --split_csv Elder-trainval/split_labels_train.csv \
  --out_dir splits/elder_cv5_label3_seed3407 \
  --label_col label3 \
  --seed 3407 \
  --overwrite
```

当前默认按三分类 `label3` 分层，二分类和三分类共用同一套 fold。

## 7. 运行 5fold * 20seed 训练

脚本里使用 20 个随机 seed：

```text
3407 42 2026 17 29 73 101 233 521 777
1009 1234 1601 1997 2701 4099 5051 6067 7411 9001
```

二分类训练：

```bash
cd submission-final

bash scripts/run_chunkfull_cv5x20_binary.sh
```

三分类训练：

```bash
cd submission-final

bash scripts/run_chunkfull_cv5x20_ternary.sh
```

常用可覆盖参数：

```bash
EPOCHS=80
BATCH_SIZE=4
LR=5e-4
PATIENCE=25
LAMBDA_REG=5.0
DEVICE=cuda
```

示例：

```bash
EPOCHS=100 PATIENCE=30 BATCH_SIZE=4 bash scripts/run_chunkfull_cv5x20_ternary.sh
```

本试验训练时使用1张 a100-pcie-40gb 跑5folds*20seeds实验，参数设置为EPOCHS=100 PATIENCE=30，

## 8. checkpoint 和 log 保存位置

二分类 checkpoint：

```text
checkpoints/cv5x20_chunkfull/binary/elder_chunkfull_c4t128_cv5x20_fold{fold}_seed{seed}/
```

三分类 checkpoint：

```text
checkpoints/cv5x20_chunkfull/ternary/elder_chunkfull_c4t128_cv5x20_fold{fold}_seed{seed}/
```

log 路径：

```text
logs/cv5x20_chunkfull/binary/elder_chunkfull_c4t128_cv5x20_fold{fold}_seed{seed}/
logs/cv5x20_chunkfull/ternary/elder_chunkfull_c4t128_cv5x20_fold{fold}_seed{seed}/
```

每个 fold/seed 都有独立文件夹，便于后续统计和选择模型。

### 全量训练集固定 epoch 训练

本研究还进行了一些补充实验。在完成常规 train/val 划分实验后，考虑到 Elder 数据量较小，从训练集中再划分验证集会进一步减少可用于参数更新的数据，并可能带来更大的划分波动，因此后续也尝试过使用训练集全量训练模型，并在若干固定 epoch 快照上进行结果验证。候选 epoch 包括 `10, 13, 17, 18, 24, 26, 37, 48, 52, 59`。在这些实验中，单模型二分类表现较好的组合包括 `seed=1009, epoch=17`、`seed=777, epoch=18`等；单模型三分类表现较好的组合包括 `seed=2701, epoch=37`、`seed=3583, epoch=52`等。

如果不再从训练集中划分验证集，而是直接使用 `Elder-trainval/split_labels_train.csv` 中的全部训练样本进行参数更新，可以使用 `scripts/run_chunkfull_fulltrain.sh`。

二分类示例：

```bash
cd submission-final

TASK=binary TRAIN_SEEDS='1009' EPOCHS=17 bash scripts/run_chunkfull_fulltrain.sh
```

三分类示例：

```bash
cd submission-final

TASK=ternary TRAIN_SEEDS='2701' EPOCHS=37 bash scripts/run_chunkfull_fulltrain.sh
```

## 9. Qwen PHQ-9 后处理校正

`qwen_phq9_correction.py` 会读取：

```text
predict/binary.csv
predict/ternary.csv
Elder-trainval/descriptions.csv
Elder-test/sample_transcriptions_qwen_asr.csv
```

并调用本地：

```text
models/Qwen2.5-72B-Instruct-AWQ
```

本试验使用一张 nvidia_h100 对模型输出的phq9值进行标签校正 

如果本地还没有该模型，可以在 `submission-final` 根目录下拉取：

```bash
cd submission-final

hf download Qwen/Qwen2.5-72B-Instruct-AWQ \
  --local-dir models/Qwen2.5-72B-Instruct-AWQ
```

如果所在网络访问 HuggingFace 较慢，可以先配置镜像后再执行同一条下载命令：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf download Qwen/Qwen2.5-72B-Instruct-AWQ \
  --local-dir models/Qwen2.5-72B-Instruct-AWQ
```

每个 subject 生成 10 次 PHQ-9 预测，解析 raw text 中的 `phq9`，然后计算：

```text
corrected_phq9 = 0.9 * mean(qwen_parsed_phq9) + 0.1 * original_phq9_pred
```

先 dry-run 检查 prompt：

```bash
cd submission-final

python qwen_phq9_correction.py \
  --dry_run \
  --limit 1 \
  --local_files_only
```

正式运行：

```bash
CUDA_VISIBLE_DEVICES=0 python \
  qwen_phq9_correction.py \
  --local_files_only \
  --n_samples 10 \
  --temperature 0.3 \
  --top_p 0.9
```
