# SAPNet

This repository contains the code for **SAPNet: Semantic-Affective Prompting with MLLMs for Incomplete Multimodal Learning**. It includes semantic-affective prompt generation, prompt feature extraction, and two-stage model training.

## Supported Datasets

- CMU-MOSI
- CMU-MOSEI
- IEMOCAP, four-class or six-class labels

The default dataset layout is:

```text
dataset/
  CMU-MOSI/
  CMU-MOSEI/
  IEMOCAP/
```

Dataset paths can be adjusted in `config.py`.

## Main Entry Points

- `train.py`: train and evaluate the model.
- `prompts/generate_prompts.py`: generate semantic prompts for modality conditions.
- `prompts/extract_features.py`: encode generated prompts into `.npy` feature files.
- `scripts/`: example shell scripts for common training runs.

## Training

```bash
python -u train.py \
  --dataset CMU-MOSI \
  --audio-feature wav2vec-large-c-UTT \
  --text-feature deberta-large-4-UTT \
  --video-feature manet_UTT \
  --prompt-feature auto \
  --test_condition atv \
  --batch-size 32 \
  --epochs 300 \
  --stage_epoch 150 \
  --gpu 0
```

The training pipeline uses:

1. First stage: train audio, text, and visual experts on complete modalities.
2. Expert selection: select the best first-stage expert states on the validation split.
3. Second stage: train the fused predictor under the requested modality condition.
4. Model selection: select the best epoch using the validation split.
5. Final report: evaluate the selected model on the test split.

Prompt features are aligned with multimodal representations only in the second stage.

## Prompt Generation

Prompt generation uses four MSA-specific stages:

```text
STATE -> EVIDENCE -> INFERENCE -> PROMPT
```

`STATE` records the modality availability, `EVIDENCE` extracts affective cues from available modalities, `INFERENCE` estimates sentiment tendency and cross-modal relation, and `PROMPT` produces the final text used for prompt feature extraction. Multiple candidates can be generated for a selected stage and judged automatically.

```bash
python -u prompts/generate_prompts.py \
  --dataset CMU-MOSI \
  --split train \
  --condition atv \
  --output-path prompt_outputs/train_audio_text_visual_fixed.jsonl \
  --raw-data-dir /path/to/raw/videos \
  --model-path /path/to/omni-model \
  --num-candidates 3 \
  --candidate-stage PROMPT
```

For text-only prompt generation, `--raw-data-dir` is not required. For audio or visual conditions, provide the dataset raw video directory.

```bash
python -u prompts/generate_prompts.py \
  --dataset IEMOCAP \
  --iemocap-classes 4 \
  --split train \
  --condition t \
  --output-path prompt_outputs/iemocap4_train_text_fixed.jsonl \
  --model-path /path/to/omni-model
```

## Prompt Feature Extraction

```bash
python -u prompts/extract_features.py \
  --dataset CMU-MOSI \
  --prompt-dir prompt_outputs \
  --conditions a t v at av tv atv \
  --model-path /path/to/deberta-large \
  --batch-size 32 \
  --max-length 48 \
  --pooling mean
```

## Repository Structure

```text
sapnet/
  datasets.py
  losses.py
  model.py
  modalities.py
  utils.py
  modules/
    attention.py
config.py
train.py
prompts/
  generate_prompts.py
  extract_features.py
scripts/
```
