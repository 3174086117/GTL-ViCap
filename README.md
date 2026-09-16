# GTL-ViCap: Global-to-Local Video Captioning

## Overview

GTL-ViCap is a global-to-local video captioning method built on CLIP4Caption. It augments the original CLIP video representation with three complementary semantic priors:

- **Topic feature:** provides high-level global semantics retrieved from a training-free topic bank built from CLIP caption embeddings.
- **Noun feature:** provides entity and object information learned by a video-to-noun alignment module.
- **Verb feature:** provides action information learned by a video-to-verb alignment module.

The final visual input is ordered as follows:

```text
[Topic Token, Noun Token, Verb Token, 17 CLIP Frame Tokens]
                              |
                              v
          2-layer Transformer Video Encoder
                              |
                              v
          2-layer Transformer Text Decoder
                              |
                              v
                       Video Caption
```

The complete workflow is:

```text
Raw videos --> CLIP frame features --+--> noun alignment --> noun priors ----+
                                     +--> verb alignment --> verb priors ----+
Training captions --> CLIP text -----+--> topic aggregation --> topic bank --+
Original CLIP frame features --------------------------------------------->GTL-ViCap training
                                                           
```

> **Implementation note:** this repository preserves its existing training-free topic aggregation algorithm: seeded random center selection, nearest-center assignment, and inverse-distance weighted aggregation. This implementation is not DPC-KNN.

## Repository Layout

```text
clip4caption-main/
├── Alignment/
│   ├── alignment_model.py       # Noun/verb alignment training
│   ├── prior_generation.py      # Aligned video-prior generation
│   └── Thematic_retrieval.py    # Video-to-topic Top-K retrieval
├── Cluster_Topic/
│   └── Cluster.py               # Caption encoding and topic aggregation
├── feature_extractor/           # CLIP4Clip frame-feature extraction
├── dataloaders/                 # MSVD and MSR-VTT data loaders
├── modules/                     # CLIP4Caption/UniVL encoder and decoder
├── scripts/                     # Linux Bash reproduction scripts
├── dataset/                     # Dataset metadata and extracted features
├── artifacts/                   # Generated checkpoints, priors, and results
└── train.py                     # Final caption-model training and evaluation
```

## Environment Setup

The recommended environment is Linux with an NVIDIA GPU, CUDA, Python 3.6.9, and PyTorch 1.10.2. Newer Python or PyTorch releases may require additional compatibility changes.

```bash
conda create -n gtl-vicap python=3.6.9 -y
conda activate gtl-vicap

pip install torch==1.10.2 torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install tqdm boto3 requests pandas scipy matplotlib nltk pickle5 opencv-python==4.5.5.62
pip install git+https://github.com/Maluuba/nlg-eval.git@master
pip install pycocoevalcap
```

The repository uses the local `CLIP/` package to load OpenAI CLIP. The first alignment or topic-building run may download the `ViT-B/32` checkpoint.

### Prepare BERT

```bash
mkdir -p modules/bert-base-uncased
cd modules/bert-base-uncased
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt
mv bert-base-uncased-vocab.txt vocab.txt
wget https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz
tar -xvf bert-base-uncased.tar.gz
rm bert-base-uncased.tar.gz
cd ../..
```

### Prepare the UniVL Initialization Checkpoint

Download the pretrained UniVL checkpoint from the [UniVL releases](https://github.com/microsoft/UniVL/releases/tag/v0) and place it at:

```text
weight/univl.pretrained.bin
```

## Data Preparation

This project supports MSVD and MSR-VTT. Arrange the datasets as follows:

```text
dataset/
├── MSVD/
│   ├── raw/                              # Raw videos, needed only for extraction
│   ├── captions/youtube_mapping.txt
│   ├── raw-captions_mapped.pkl
│   ├── train_list_mapping.txt
│   ├── val_list_mapping.txt
│   ├── test_list_mapping.txt
│   ├── MSVD/new_V_N_MSVD.json           # caption, nouns, verbs, video_id
│   └── MSVD_CLIP4Clip_features.pickle
└── MSRVTT/
    ├── raw/                              # Raw videos, needed only for extraction
    ├── MSRVTT_data.json
    ├── msrvtt.csv
    ├── captions.json                    # Training-caption list
    ├── V_N_MSRVTT.json                  # caption, nouns, verbs, video_id
    └── MSRVTT_CLIP4Clip_features.pickle
```

The noun/verb annotation files must be JSON lists. Each entry must contain at least the following fields:

```json
{
  "video_id": "vid1",
  "caption": "a squirrel is eating a peanut",
  "nouns": "squirrel peanut",
  "verbs": "eating"
}
```

All raw video, noun, verb, and topic representations must be 512-dimensional.

## Reproduction

Run all commands from the repository root unless stated otherwise.

### Step 1: Extract CLIP Video Features

Place the pretrained CLIP4Clip checkpoints at:

```text
feature_extractor/pretrained_clip4clip/msvd/pytorch_model.bin
feature_extractor/pretrained_clip4clip/msrvtt/pytorch_model.bin
```

Extract 17 uniformly sampled frame features per video:

```bash
cd feature_extractor

python clip_feature_extractor.py \
  --dataset_type msvd \
  --dataset_dir ../dataset \
  --save_dir ../extracted_feats \
  --max_frames 17

python clip_feature_extractor.py \
  --dataset_type msrvtt \
  --dataset_dir ../dataset \
  --save_dir ../extracted_feats \
  --max_frames 17

cd ..
```

Move the generated files into the dataset layout shown above, or pass their original paths to the later commands.

### Step 2: Train the Noun and Verb Alignment Modules

Alignment training freezes the CLIP image and text encoders. Only the temporal Transformer and projection layers are optimized using a symmetric video-to-text and text-to-video InfoNCE loss.

Train both branches for MSVD and MSR-VTT:

```bash
bash scripts/train_alignment.sh MSVD
bash scripts/train_alignment.sh MSRVTT
```

Expected checkpoints:

```text
artifacts/alignment/msvd/msvd_nouns_alignment_best.pth
artifacts/alignment/msvd/msvd_verbs_alignment_best.pth
artifacts/alignment/msrvtt/msrvtt_nouns_alignment_best.pth
artifacts/alignment/msrvtt/msrvtt_verbs_alignment_best.pth
```

To train one branch manually:

```bash
python Alignment/alignment_model.py \
  --dataset MSVD \
  --align_type nouns \
  --features_path dataset/MSVD/MSVD_CLIP4Clip_features.pickle \
  --annotations_path dataset/MSVD/MSVD/new_V_N_MSVD.json \
  --split_dir dataset/MSVD \
  --output_dir artifacts/alignment/msvd \
  --epochs 10 \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --weight_decay 0.01 \
  --max_frames 17 \
  --device auto
```

### Step 3: Generate Noun and Verb Priors

Use the trained alignment branches to generate one 512-dimensional noun feature and one 512-dimensional verb feature for every video:

```bash
bash scripts/generate_priors.sh MSVD
bash scripts/generate_priors.sh MSRVTT
```

Expected outputs:

```text
artifacts/priors/msvd_nouns.pkl
artifacts/priors/msvd_verbs.pkl
artifacts/priors/msrvtt_nouns.pkl
artifacts/priors/msrvtt_verbs.pkl
```

The equivalent direct command is:

```bash
python Alignment/prior_generation.py \
  --model_path artifacts/alignment/msvd/msvd_nouns_alignment_best.pth \
  --features_path dataset/MSVD/MSVD_CLIP4Clip_features.pickle \
  --save_path artifacts/priors/msvd_nouns.pkl \
  --max_frames 17 \
  --batch_size 128 \
  --device auto
```

### Step 4: Build the Training-Free Topic Bank

Build 1,000 topic features from the training captions:

```bash
bash scripts/build_topics.sh MSVD
bash scripts/build_topics.sh MSRVTT
```

Expected outputs:

```text
artifacts/topics/msvd_topics_1000.pt
artifacts/topics/msrvtt_topics_1000.pt
```

The equivalent direct command for MSVD is:

```bash
python Cluster_Topic/Cluster.py \
  --dataset MSVD \
  --captions_path dataset/MSVD/MSVD/new_V_N_MSVD.json \
  --train_split_path dataset/MSVD/train_list_mapping.txt \
  --target_topic_num 1000 \
  --output_path artifacts/topics/msvd_topics_1000.pt \
  --batch_size 256 \
  --block_size 2000 \
  --seed 42 \
  --device auto
```

This module does not use gradient-based training. It encodes captions with CLIP, selects aggregation centers using a fixed random seed, assigns each caption embedding to its nearest center, and performs inverse-distance weighted aggregation. During caption-model training, the system retrieves the Top-K topic features using cosine similarity between the topic bank and the masked mean of the original CLIP frame features.

### Step 5: Train the Final Caption Model

```bash
bash scripts/train_msvd.sh
bash scripts/train_msrvtt.sh
```

The default configuration follows the accompanying manuscript:

- 17 original CLIP frame tokens
- 1 topic token
- 1 noun token
- 1 verb token
- Final visual sequence length of 20
- Maximum caption length of 48
- 25 training epochs
- Initial learning rate of `1e-5`
- Warmup proportion of `0.1`

Training outputs are written to:

```text
artifacts/caption/msvd/
artifacts/caption/msrvtt/
```

Each output directory contains epoch checkpoints, logs, an epoch-level metrics CSV, and training plots. The checkpoint with the best validation CIDEr score is evaluated on the test split at the end of training.

### Step 6: Evaluate a Checkpoint

Pass the checkpoint path as the first script argument:

```bash
bash scripts/eval_msvd.sh artifacts/caption/msvd/pytorch_model.bin.8
bash scripts/eval_msrvtt.sh artifacts/caption/msrvtt/pytorch_model.bin.0
```

Evaluation reports BLEU@4, METEOR, ROUGE-L, and CIDEr. Caption generation uses beam search with a beam width of 5.

## Reported Results

The following results are reported by the accompanying manuscript. They were not reproduced as part of this repository update.

| Dataset | Method | BLEU@4 | METEOR | ROUGE-L | CIDEr |
|---|---|---:|---:|---:|---:|
| MSVD | CLIP4Caption baseline | 56.3 | 38.7 | 75.9 | 104.9 |
| MSVD | GTL-ViCap | **62.2** | **41.8** | **79.2** | **121.6** |
| MSR-VTT | CLIP4Caption baseline | 46.1 | 30.7 | 63.7 | 57.7 |
| MSR-VTT | GTL-ViCap | **48.8** | **31.9** | **65.9** | **64.3** |

### Ablation Study

| Noun | Verb | Topic | MSVD B@4 | MSVD ROUGE-L | MSVD METEOR | MSVD CIDEr | MSR-VTT B@4 | MSR-VTT ROUGE-L | MSR-VTT METEOR | MSR-VTT CIDEr |
|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  |  |  | 56.3 | 75.9 | 38.7 | 104.9 | 46.1 | 63.7 | 30.7 | 57.7 |
| ✓ |  |  | 61.6 | 78.3 | 41.2 | 120.7 | 50.0 | 65.9 | 32.2 | 62.6 |
|  | ✓ |  | 60.3 | 78.3 | 41.2 | 116.8 | 48.4 | 65.4 | 31.7 | 61.5 |
|  |  | ✓ | 60.1 | 78.0 | 40.8 | 117.5 | 48.5 | 65.4 | 31.9 | 61.9 |
|  | ✓ | ✓ | 61.1 | 78.4 | 41.3 | 118.6 | 48.8 | 65.4 | 31.7 | 61.5 |
| ✓ | ✓ |  | 60.9 | 78.8 | 41.2 | 120.3 | 48.6 | 65.8 | 32.0 | 63.9 |
| ✓ |  | ✓ | 62.3 | 78.5 | 41.7 | 121.0 | 48.1 | 65.4 | 31.8 | 63.1 |
| ✓ | ✓ | ✓ | **62.2** | **79.2** | **41.8** | **121.6** | **48.8** | **65.9** | **31.9** | **64.3** |

## Large Files and Git

CLIP features, aligned priors, topic banks, and model checkpoints commonly exceed GitHub's 100 MB per-file limit. Do not commit these files through regular Git. Use Git LFS, release assets, or an external download service instead.

The repository ignores generated artifacts and common large model formats, including:

```text
weight/
artifacts/
*.pickle
*.pkl
*.pt
*.pth
*.bin
*.bin.*
```

Files already tracked by Git remain tracked even if they match these patterns.

## Acknowledgements

This repository builds on the following projects:

- [CLIP4Caption: CLIP for Video Caption](https://dl.acm.org/doi/10.1145/3474085.3479207)
- [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip)
- [UniVL](https://github.com/microsoft/UniVL)
- [OpenAI CLIP](https://github.com/openai/CLIP)

The method description follows the accompanying manuscript, *GTL-ViCap: A Global-to-Local Video Captioning Method*. The manuscript attachment does not provide complete author, venue, or DOI metadata, so those details are intentionally not invented here.
