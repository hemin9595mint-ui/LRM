# A Physics-Guided Framework for Underwater Video Enhancement in Aquaculture Environments

This repository provides the dataset resources, evaluation utilities, reproducibility information, and supplementary materials for our work on **physics-guided weakly supervised underwater video enhancement for aquaculture environments**.

## 1. IWUV Dataset

We introduce the **Inland Water Underwater Video (IWUV) dataset**, which was collected in real aquaculture environments and contains diverse underwater degradations, including turbidity, non-uniform illumination, color distortion, suspended-particle interference, occlusion, and dynamic scene changes.

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

## 2. Dataset Download

### IWUV dataset package — Google Drive

The released IWUV dataset package is available at:

**Google Drive:**  
https://drive.google.com/drive/folders/1KNclntJYwPm4SKDm6jheOOVMnkB8PWWZ?usp=sharing

### Complete video resources — Baidu Netdisk

The complete underwater video resources are available at:

**Baidu Netdisk:**  
https://pan.baidu.com/s/1aDUMFCk1-qVB29uUQ8QeGg

**Password:** `1234`

If one download service is temporarily unavailable, please try the alternative source or open an issue in this repository.

## 3. Training and Evaluation Protocol

The proposed model is trained using IWUV training sequences together with an independent **unpaired clean-domain reference set**.

- **840 high-quality UIEB reference images** are used only for unpaired clean-domain guidance.
- **475 clean LOL images** are additionally used as clean-domain references.
- These clean-domain images are not paired with IWUV sequences.
- The remaining **50 UIEB image pairs** are reserved exclusively for full-reference evaluation.
- The **UIEB Challenging-60** images are used only for no-reference evaluation.
- MVK is used for cross-dataset video evaluation without adaptation.
- DeepFish is used for zero-shot downstream fish-detection evaluation without enhancement-model fine-tuning.

No clean reference images are required during inference.

## 4. Software Environment

The main experimental environment is:

- **OS:** Ubuntu 22.04
- **Python:** 3.12
- **PyTorch:** 2.8.0
- **CUDA:** 12.8
- **GPU:** NVIDIA Tesla V100, 32 GB

The exact software dependencies and package versions required for reproduction should be installed from:

```bash
pip install -r requirements.txt
```

> **Before final resubmission:** ensure that `requirements.txt` is included in the repository and records the exact versions used in the experiments.

## 5. Main Training Settings

The main training settings reported in the manuscript are:

- Input sequence length: **5 frames**
- Crop size: **256 × 256**
- Batch size: **8**
- Optimizer: **Adam**
- Initial learning rate: **2 × 10^-4**
- Training epochs: **200**
- Random seeds: **4, 42, 123, 3407, 2024**
- Model selection: best checkpoint on the validation set
- Reported results for our model: mean ± standard deviation across five independent runs

### Loss weights

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

The final hyperparameter configuration was selected using the **validation set only**, considering both enhancement quality and temporal consistency. The test sets were not used for hyperparameter tuning or checkpoint selection.

> **Reviewer-requested reproducibility record:** before final resubmission, add the actual **candidate values/ranges considered** and the **number of validation trials/configurations** here (or in a dedicated `HYPERPARAMETERS.md` file). These values should reflect the experiments that were actually performed.

## 6. Evaluation Metrics and Scripts

During revision, we re-examined the evaluation pipeline and provide corrected implementations for the underwater no-reference image-quality metrics used in the experiments.

The corrected evaluation utilities include:

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

is retained only to document the previous implementation and should **not** be used to reproduce the final reported results.

### Basic evaluation workflow

1. Install the required environment using `requirements.txt`.
2. Prepare enhanced images in a separate output directory while preserving consistent file naming.
3. Use the corrected scripts under `right-Underwater-image-evaluation-metrics/`.
4. Apply the same preprocessing and metric implementation to all compared methods.
5. Report the average metric values over the corresponding evaluation set.

For paired UIEB evaluation, PSNR, SSIM, and LPIPS are computed between enhanced images and their aligned reference images. For video evaluation, the manuscript additionally reports VSFA, FastVQA, DOVER, and warping error (WE) using the corresponding official implementations/settings described in the paper.

## 7. Repository Checklist for Reproducibility

Before final paper resubmission, the repository should provide public and stable access to the following items:

- [x] Public IWUV dataset download link
- [x] Complete video-resource download link
- [x] Corrected underwater evaluation metric implementations
- [ ] `requirements.txt` with exact dependency versions
- [ ] Training/inference code, if released with the revision
- [ ] Evaluation instructions for all released scripts
- [ ] Hyperparameter candidate ranges and number of validation trials
- [ ] Final directory/file organization documented in this README

This checklist is included to make the released resources easy to verify and reproduce.

## 8. Suggested Dataset Organization

A recommended local organization is:

```text
IWUV/
├── train/
├── val/
├── sequence_test/
└── heldout_video_test/
```

Please keep the provided split organization unchanged when reproducing the reported experiments so that source-video separation is preserved.

## 9. Citation

If you use the IWUV dataset, evaluation utilities, or this work in your research, please cite the corresponding paper.

```bibtex
@article{xxx,
  title   = {A Physics-Guided Framework for Underwater Video Enhancement in Aquaculture Environments},
  author  = {Min He and Dongfang Li and Tieli Lyu and Zhiyuan Chen and Zhu Liu and Jie Hu and Maohua Xiao},
  journal = {Pattern Recognition},
  year    = {2026}
}
```

> Please replace the BibTeX entry with the final publication information once available.

## 10. Contact

For questions about the IWUV dataset, evaluation scripts, or reproducibility materials, please open an issue in this repository or contact the corresponding author.
