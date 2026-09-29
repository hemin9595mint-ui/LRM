# A Physics-Guided Framework for Underwater Video Enhancement in Aquaculture Environments

This repository provides the dataset resources, evaluation utilities, reproducibility information, and supplementary materials for our work on **physics-guided weakly supervised underwater video enhancement for aquaculture environments**.

## IWUV Dataset

We introduce the **Inland Water Underwater Video (IWUV) dataset**, collected in real aquaculture environments with diverse underwater degradations, including turbidity, non-uniform illumination, color distortion, suspended-particle interference, occlusion, and dynamic scene changes.

### Dataset protocol

The IWUV data are partitioned at the **source-video level before sequence construction** to avoid overlap between model-development and evaluation subsets.

- Each input sequence contains **5 consecutive frames**.
- A total of **1,315 five-frame sequences** are used for sequence-level experiments:
  - **1,060 training sequences**
  - **150 validation sequences**
  - **105 sequence-level test sequences**
- The source videos used to construct the training, validation, and sequence-level test subsets are mutually exclusive.
- An additional **114 complete videos** from different acquisition sites and recording periods are reserved for **independent held-out video-level evaluation**.
- Frames from these 114 held-out videos are not used for training, validation, checkpoint selection, or hyperparameter tuning.

## Dataset Download

### IWUV dataset package — Google Drive

https://drive.google.com/drive/folders/1KNclntJYwPm4SKDm6jheOOVMnkB8PWWZ?usp=sharing

### Complete video resources — Baidu Netdisk

https://pan.baidu.com/s/1aDUMFCk1-qVB29uUQ8QeGg

Password: `1234`

## Training and Evaluation Protocol

The proposed model is trained using IWUV training sequences together with an independent **unpaired clean-domain reference set**.

- **840 high-quality UIEB reference images** are used only for unpaired clean-domain guidance.
- **475 clean LOL images** are additionally used as clean-domain references.
- The clean-domain images are not paired with IWUV sequences.
- The remaining **50 UIEB image pairs** are reserved exclusively for full-reference evaluation.
- The **UIEB Challenging-60** images are used only for no-reference evaluation.
- MVK is used for cross-dataset video evaluation without adaptation.
- DeepFish is used for zero-shot downstream fish-detection evaluation without enhancement-model fine-tuning.

No clean reference images are required during inference.

## Software Environment

The main experimental environment is:

- **OS:** Ubuntu 22.04
- **Python:** 3.12
- **PyTorch:** 2.8.0
- **CUDA:** 12.8
- **GPU:** NVIDIA Tesla V100, 32 GB

Install the exact software dependencies from the released environment file:

```bash
pip install -r requirements.txt
```

## Main Training Settings

- Input sequence length: **5 frames**
- Crop size: **256 × 256**
- Batch size: **8**
- Optimizer: **Adam**
- Initial learning rate: **2 × 10^-4**
- Training epochs: **200**
- Random seeds: **4, 42, 123, 3407, 2024**
- Model selection: best checkpoint on the validation set
- Reported results for our model: mean ± standard deviation across five independent runs

### Final loss weights

| Hyperparameter | Final value |
|---|---:|
| `lambda_deg` | 1.0 |
| `lambda_cdg` | 1.0 |
| `lambda_A` | 0.01 |
| `lambda_N` | 0.01 |
| `lambda_temp` | 0.01 |
| `lambda_adv` | 0.01 |
| `lambda_col` | 1.0 |
| `lambda_sty` | 0.01 |
| `lambda_str` | 0.1 |

The final hyperparameter configuration was selected using the **validation set only**, considering both frame-level enhancement quality and temporal consistency. The test sets were not used for hyperparameter tuning or checkpoint selection.

Detailed candidate ranges, validation trials, and the final selection record are provided in `HYPERPARAMETERS.md`.

## Evaluation Metrics and Scripts

The repository provides corrected implementations for the underwater no-reference image-quality metrics used in the experiments:

- UCIQE
- UIQM
- UICM
- UISM
- UIConM

The corrected implementations are provided in:

```text
right-Underwater-image-evaluation-metrics/
```

The file:

```text
underwater_metrics_ERROR.py
```

is retained only for documenting the previous implementation and should **not** be used to reproduce the final reported results.

### Basic evaluation workflow

1. Install the required environment using `requirements.txt`.
2. Prepare enhanced images in a separate output directory while preserving consistent file naming.
3. Use the corrected scripts under `right-Underwater-image-evaluation-metrics/`.
4. Apply the same preprocessing and metric implementation to all compared methods.
5. Report the average metric values over the corresponding evaluation set.

For paired UIEB evaluation, PSNR, SSIM, and LPIPS are computed between enhanced images and their aligned reference images. For video evaluation, the manuscript additionally reports VSFA, FastVQA, DOVER, and warping error (WE) using the corresponding official implementations/settings described in the paper.

## Suggested Dataset Organization

```text
IWUV/
├── train/
├── val/
├── sequence_test/
└── heldout_video_test/
```

Please keep the provided split organization unchanged when reproducing the reported experiments so that source-video separation is preserved.

## Reproducibility Resources

The repository provides:

- IWUV dataset download links
- complete underwater video resources
- corrected evaluation metric implementations
- exact software dependencies in `requirements.txt`
- hyperparameter-selection record in `HYPERPARAMETERS.md`
- usage and evaluation instructions in this README

