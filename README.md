[Read our Medium blog post](https://medium.com/@eldadc1/what-we-learned-while-training-a-conformer-based-speech-recognition-model-01533474c0a7)
# README for Conformer Model  
## Training and Evaluation Setup

---

## Setup

### 1. Create a Conda Environment on Linux
```bash
conda create -n conformer0 python=3.10 sox -c conda-forge
conda activate conformer0
```

**Notes:**
- While not strictly required, using Conda is advised for preventing conflicts and keeping your environments clean.
- The `sox` package supplies the necessary torchaudio backend for Linux.
- Python 3.10 is required.

### 2. Install necessary packages
```bash
pip install -r requirements.txt
```

### 3. Download the LibriSpeech dataset (if missing)
```bash
python download-librispeech.py
```

**Notes:**
- By default, this downloads 6 splits of the dataset: `train-clean-100`, `train-clean-360`, `train-other-500`, `dev-clean`, `test-clean`, and `test-other` into `./dataset/LibriSpeech`.
- The script uses a European server for faster downloads and asks for confirmation before proceeding.
- Additional flags can customize which parts to download — not necessarily tied to this project.

---

## Train and Evaluate

### 1. (Optional) Change HuggingFace default cache directory
For example:
```bash
setenv HF_HOME /home/yandex/APDL2425a/group_8/bin/HuggingFace-cache
```
This might not be necessary for everyone, but the default cache directory might not be sufficient to download the pre-trained language model.

### 2. Run training and evaluation
```bash
python train.py hparams/conformer_small.yaml --data_folder datasets/LibriSpeech/
```

**Notes:**
- The `conformer_small.yaml` file defines the model architecture and hyperparameters, based on a recipe from the SpeechBrain repo (commit: `d9fb58f56`).
- We modified batch size (from 16 → 12) and turned dynamic batching off due to memory constraints.
- You can compare with `hparams/original_conformer_small.yaml` if desired.
- `train.py` handles both training and evaluation end-to-end and includes a checkpoint mechanism to resume training seamlessly.
- If the fully-trained model is available in the directory, running the command will skip straight to evaluation.
- Output includes training progress, learning rate, losses, validation accuracy, and final WER (Word Error Rate) on test sets.
- Results are stored under `./results/conformer_small/7775`. Check `save/` for model files and logs like:
  - `train_log.txt`
  - `wer_test-clean.txt`
  - `wer_test-other.txt`

---

## Additional Contents

### 1. `analyze_training.py`
Analyzes training logs (stdout from `train.py`) to create plots such as training/validation loss, and accuracy.

- Supports multiple logs as input (from successive checkpointed runs).
- Requires all logs in correct order for complete training history.
- Example:
```bash
python analyze_training.py results/awesome.out results/awesome2.out results/awesome3.out results/awesome4.out results/awesome5.out --epochs 110
```
- Use `--help` to see all arguments.

### 2. `img/`
Contains relevant plots and media, which can be reproduced using the logs and `analyze_training.py`.

### 3. `extract_recordings.py`
Randomly extracts sample audio recordings from the LibriSpeech dataset and saves them as MP3 files (originals are FLAC).

- Use `--help` to view options and flags.

### 4. `sample-recordings/`
Contains sample audio files from various dataset splits. These were extracted using `extract_recordings.py`. You can run the script again for different random samples.

### 5. `results/`
Includes:
- Model checkpoints
- Training/evaluation logs
- Aggregated metrics

These logs span multiple runs (due to resource limitations) and demonstrate the usefulness of the checkpointing system.

---

## Submitters

**By:** Sharyn Sircovich Sassun and Eldad Cohen  
**Course:** Advanced Topics in Audio Processing using Deep Learning — Final Project  
**Lecturer:** Tal Rosenwein  
**Semester:** 2025A  
**Submitted:** April 2025
