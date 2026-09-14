<div align="center">

# Hub-Spectral Activation of Latent Multimodal Knowledge

**HSA · Recovering and activating cross-modal relations in frozen representations**

Ying Guo, Haidong Chen, Linrui Xu, Xiaohao Liu, Chuancheng Shi, Canran Xiao, Dan Zhang, Fei Shen, Li Shen, Tat-Seng Chua

[Overview](#overview) · [Method](#method) · [Results](#main-results) · [Installation](#installation) · [Data](#data-preparation) · [Usage](#usage)

</div>

## Overview

Hub-based multimodal binding connects modalities through a shared hub. What cross-modal knowledge can be recovered between two modalities that were never directly trained together?

**Hub-Spectral Activation (HSA)** recovers and activates the hub-readable component of latent multimodal knowledge from the second-order statistics of two observed hub edges. It identifies paired spectral carriers and turns their coordinates into scores for bidirectional retrieval and prototype classification.

![From multimodal training to Hub-Spectral Activation](assets/overview.png)

### Key features

- **Frozen representations:** no backbone updates, target-pair supervision, or gradient optimization for HSA.
- **Closed-form relation recovery:** compose and standardize the statistics of the two observed hub edges.
- **Paired spectral carriers:** combine reliability-weighted matching evidence with source-gated candidate resolution.
- **Two task readouts:** bidirectional retrieval and B-to-A prototype classification.

## Method

![HSA method overview](assets/method.png)

1. **Recover the relation.** Compose hub-edge cross-covariances through the regularized hub covariance, then whiten the target spaces.
2. **Identify paired carriers.** Extract spectral directions and project the frozen target representations into carrier coordinates.
3. **Construct the score.** Combine the carrier score with candidate-resolution evidence, using a source-derived reliability gate and mismatch scales.
4. **Read out the task.** Rank retrieval candidates or score a common bank of class prototypes.

$$s_{\mathrm{HSA}}=s_C/\tau_C+g_Rs_R/\tau_R.$$

All fitted quantities use source hub-edge features. Prototype classification uses class labels to construct the A-side prototype bank; test labels are used only for evaluation.

## Main results

Results reported in the manuscript:

| Backbone | Retrieval relations | Frozen cosine R@10 | HSA R@10 | Classification relations | Frozen cosine macro Top-1 | HSA macro Top-1 |
|---|---:|---:|---:|---:|---:|---:|
| ImageBind | 9 | 10.80% | **24.90%** | 6 | 16.31% | **44.01%** |
| LanguageBind | 10 | 25.00% | **36.79%** | 5 | 44.25% | **62.53%** |
| All | 19 | 18.27% | **31.15%** | 11 | 29.01% | **52.43%** |

Retrieval averages Recall@10 across both directions and then equally across relations. Classification averages Top-1 accuracy equally across observed classes within each relation and then equally across relations. The task-specific source features and splits follow the manuscript.

## Installation

Python 3.10 or later is required. CPU evaluation is supported; a CUDA-enabled PyTorch installation can accelerate retrieval scoring.

```bash
git clone https://github.com/Luo1Yan/HSA.git
cd HSA
python -m pip install -r requirements.txt
```

This implementation operates on pre-extracted features from frozen [ImageBind](https://github.com/facebookresearch/ImageBind) and [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) encoders. Backbone training is outside the evaluation entry point.

## Data preparation

Main experiments cover VGGSound, UCF101, NYUv2, TartanRGBT, Ego4D, BatVision, MAVD, UTD-MHAD, Caltech Aerial RGBT, and MSR-VTT. Dataset contents and encoder checkpoints are kept outside Git.

**Prepared feature archives are not yet publicly hosted.** With local archives in the format below, prepare and validate them using:

```bash
bash datasets.sh --from /path/to/prepared/features
bash datasets.sh --check
```

The script validates all 30 main-experiment archives before copying them. It preserves existing files and stops on conflicting contents. `bash datasets.sh --help` describes its options.

### Feature layout

```text
datasets/
├── retrieval/
│   ├── imagebind/<relation>.npz
│   └── languagebind/<relation>.npz
└── classification/
    ├── imagebind/<relation>.npz
    └── languagebind/<relation>.npz
```

Use `python run.py --list` for the exact 19 retrieval and 11 classification relation IDs. Each archive contains plain NumPy arrays; pickle objects are unsupported.

| Array | Shape | Meaning |
|---|---|---|
| `a_train`, `h_a_train` | N × d, N × h | Paired A-H source-edge features |
| `b_train`, `h_b_train` | N × d, N × h | Paired B-H source-edge features |
| `a_test`, `b_test` | M × d each | Retrieval gallery/query features, with corresponding rows |
| `positive_labels` | M (optional) | Retrieval positive-group labels; omit for instance matching |
| `prototypes` | C × d | Classification A-side prototype bank |
| `queries` | M × d | Classification B-side test queries |
| `labels` | M integers | Query labels indexing prototype rows, from 0 to C−1 |

Retrieval archives need the test arrays; classification archives need the prototype/query/label arrays. Source features must follow the original task's extraction and normalization protocol. Classification normalizes source rows, prototypes, and queries. Retrieval uses the supplied fitting coordinates; frozen cosine applies row normalization directly to test features.

For classification, average normalized A-side training features within each class and normalize each mean. The ImageBind VGGSound text-audio relation uses the manuscript's fixed 80-template text-prompt prototypes. Keep the original splits and prototype banks to compare with the reported results.

## Usage

### Main experiments

```bash
# Default: 19 retrieval relations and 11 prototype-classification relations
python run.py

# Retrieval only
python run.py --task retrieval --device cuda

# Prototype classification only
python run.py --task classification

# One main relation
python run.py --task classification --backbone imagebind \
  --relation depth_imu__utd --output outputs/utd.json
```

The runner evaluates **HSA and frozen cosine**. Its default inventory contains only the main retrieval and prototype-classification experiments. Results include per-relation metrics, both retrieval directions, fitted ranks, reliability gates, mismatch scales, and relation-weighted aggregates. Outputs use fractions in [0, 1]. A selected subset is averaged over that subset and reports its relation count.

The manuscript tables include additional comparison methods; this compact implementation provides the HSA and frozen-cosine paths. Mechanism analyses and ablations are outside the default runner.

### Configuration

All HSA fitting settings match the archived main implementation: trace-scaled ridge 0.5, rank threshold multiplier 2, maximum rank 256, rank-null seed 142, 256 coordinate permutations with seed 42 and quantile 0.95, and mismatch seeds 42–46. Retrieval scoring uses float32 tensors; covariance and spectral calculations retain the archived numerical conventions.

For a custom data location or scoring batch size:

```bash
python run.py --data-root /path/to/features --batch-size 128 \
  --output outputs/main.json
```

## Project structure

```text
HSA/
├── README.md
├── hsa.py             # Relation fitting, calibration, and task readouts
├── run.py             # Main experiment inventory and evaluation
├── datasets.sh        # Feature preparation and validation
├── requirements.txt
└── assets/
    ├── overview.png
    └── method.png
```

## Citation

```bibtex
@misc{guo2026hsa,
  title={Hub-Spectral Activation of Latent Multimodal Knowledge},
  author={Guo, Ying and Chen, Haidong and Xu, Linrui and Liu, Xiaohao and
          Shi, Chuancheng and Xiao, Canran and Zhang, Dan and Shen, Fei and
          Shen, Li and Chua, Tat-Seng},
  year={2026},
  note={Manuscript},
  url={https://github.com/Luo1Yan/HSA}
}
```
