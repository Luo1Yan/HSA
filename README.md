<div align="center">

# Hub-Spectral Activation of Latent Multimodal Knowledge

**HSA · Recovering and activating cross-modal relations in frozen representations**

Ying Guo, Haidong Chen, Linrui Xu, Xiaohao Liu, Chuancheng Shi, Canran Xiao, Dan Zhang, Fei Shen, Li Shen, Tat-Seng Chua

[Overview](#overview) · [Method](#method) · [Results](#main-results) · [Installation](#installation) · [Data](#data-sources-and-access) · [Usage](#a-complete-example-utd-mhad)

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

## Reproduction in three steps

The project has three public entry points. They share the experiment definitions,
sample identifiers and splits in `configs/`.

| Entry point | Responsibility |
|---|---|
| `download.py` | Acquire original datasets and pinned frozen encoder code/weights |
| `extract.py` | Prepare model inputs, extract frozen features, construct task archives |
| `run.py` | Fit HSA and evaluate retrieval or prototype classification |

HSA fitting is closed form. All backbone parameters remain frozen. Each extracted
observation is cached once and reused by the applicable retrieval and classification
experiments. Features and model weights are generated or downloaded locally.

### Installation

Use Python 3.10. The extraction dependencies include FFmpeg through `imageio-ffmpeg`;
an FFmpeg executable on `PATH` takes precedence. A CUDA GPU is recommended for extraction.
The evaluation entry point also supports CPU.

```bash
git clone https://github.com/Luo1Yan/HSA.git
cd HSA
# In an isolated Python 3.10 environment, for an NVIDIA CUDA 12.1 build:
python -m pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r environments/standard.txt
```

Choose a suitable GPU or CPU build from the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
Use `--device cpu` in the example below when a CUDA device is unavailable.

The standard environment covers the main ImageBind extraction and the original
LanguageBind protocols. The official NYUv2 classification and VGGSound-309
classification extractions use `environments/classification.txt` in a separate
Python environment. Select those protocols with `--environment classification`;
use `--environment standard` for the other extraction protocols. Both produce
archives consumed by the same `run.py`.

For those two classification protocols, activate a separate Python 3.10 environment
and install `environments/classification.txt`. It pins PyTorch 2.8.0; select a
corresponding CUDA build from the same official instructions if using a GPU.

For evaluation of existing local features, the smaller `requirements.txt` suffices.
Windows users can run the same Python commands in PowerShell.

### A complete example: UTD-MHAD

```bash
# Show original data and model sources before downloading.
python download.py --dataset utd-mhad --backbone imagebind

# Step 1: download and unpack the original data and the frozen model.
python download.py --dataset utd-mhad --backbone imagebind --fetch

# Step 2: produce both registered task archives from original RGB/depth/IMU inputs.
python extract.py --dataset utd-mhad --backbone imagebind --device cuda

# Step 3: fit HSA and score both tasks.
python run.py --dataset utd-mhad --backbone imagebind --device cuda --output outputs/utd.json
```

`--data-root` has the same meaning in all three commands and defaults to
`datasets`. `--model-root` defaults to `models`. Select an individual experiment
with `--task`, `--backbone`, and `--relation`; `--list` shows the available choices.

### Data sources and access

`download.py` prints its acquisition plan by default. `--fetch` executes the plan.
`--data-only`, `--models-only` and `--no-unpack` allow separate acquisition steps.
Public Hub repositories in this table can be downloaded without a Hugging Face
account. Downloads of the complete video datasets can be large; select one dataset
at a time and review its source and storage requirements.

| Dataset | Original-data source used by the entry point | Registered preparation |
|---|---|---|
| UTD-MHAD | [UT Dallas](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html) | RGB middle frame, depth sequence, six-axis inertial signal; odd/even subjects |
| UCF101 | [CMCL's UCF101 mirror](https://huggingface.co/datasets/NoahMartinezXiang/2013_UCF101) | Fixed 7070/3030 observations, middle-frame image and video |
| NYUv2 | [CMCL's depth mirror](https://huggingface.co/datasets/sayakpaul/nyu_depth_v2) and [official labeled data](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | Separate CMCL retrieval and official 795/654 classification protocols |
| VGGSound | [CMCL's video mirror](https://huggingface.co/datasets/Loie/VGGSound) and [test-audio mirror](https://huggingface.co/datasets/txya900619/vggsound-16k) | Fixed 100-class retrieval and 309-class classification collections |
| TartanRGBT | [Official release](https://huggingface.co/datasets/theairlabcmu/TartanRGBT) | Paired RGB/thermal/depth frames and registered eight-frame video windows |
| BatVision | [Official dataset](https://doi.org/10.57745/HYLZNL) | BatVision V2 scene CSVs, audio and inverse-depth images |
| MAVD | [Official release](https://multimodal-distill.cs.uni-freiburg.de/) | Registered drive/sample IDs; original multipart archives |
| Caltech Aerial RGBT | [Official release](https://data.caltech.edu/records/cks6g-ps927) | Synchronized ROS image messages and gyro-first IMU windows |
| MSR-VTT | [PMRL's original-data mirror](https://huggingface.co/datasets/xhLiu/MM_eval) | Fixed 6176/884 videos, captions and decoded audio |
| Ego4D | [Official access instructions](https://ego4d-data.org/docs/start-here/) | Licensed v2 video, narration and IMU files; registered IDs and temporal windows |

Ego4D requires the user's own authorized dataset access. The download entry point
prints the required files and official procedure. After acquisition, put the files
under `datasets/raw/ego4d/`; the extraction interface checks every registered input.
Dataset and checkpoint licenses remain those of their respective providers.

### Validate inputs and resume

```bash
python extract.py --dataset utd-mhad --check-inputs
python run.py --dataset utd-mhad --check-data
```

Missing observations stop extraction and produce a report in `datasets/checks/`.
The pipeline preserves the registered sample list and never replaces unavailable
observations. Rerunning extraction reuses completed observation features. Use
`--force` to regenerate them after replacing original data or model weights.

The NYUv2 and VGGSound task-specific configurations preserve their distinct
sample collections, depth conventions and prototype construction. LanguageBind's
upstream video preprocessing contains random horizontal flips; this release assigns
a fixed seed per observation so batching, task order and resumption use the same
augmentation. The historical UCF101/Tartan video caches did not record each flip,
so bitwise reconstruction of those historical video features is not certified.

### Verification scope

- The shared task builder was checked against the archived inputs for all **19
  retrieval and 11 classification** main experiments, including labels and prototypes.
- The evaluation code on those archived inputs reproduces **31.15% retrieval
  Recall@10** and **52.43% classification macro Top-1**.
- UTD-MHAD was re-extracted from the original RGB, depth and inertial archives and
  evaluated through the public entry points. HSA obtains **9.30% retrieval
  Recall@10** and **34.75% classification macro Top-1**, matching the reported results.
- The downloaded, pinned LanguageBind source was checked on eight NYUv2 observations
  for each of image, depth and text against the original retrieval cache. The official
  classification image/depth inputs and the ten-class text bank were also checked.
- Raw-input adapter pilots passed 104 checks across the 13 registered observation
  collections, covering file lookup and media/sensor preparation where applicable.
- A complete fresh acquisition and re-extraction of all ten datasets has not been
  certified. In particular, access-controlled files, availability of fixed video
  IDs, decoder versions and stochastic legacy video preprocessing matter.

The runner evaluates **HSA and frozen cosine**. Results contain per-relation
metrics, retrieval directions, fitted ranks, reliability gates, mismatch scales
and equal-weight relation aggregates. Numerical outputs are fractions in [0, 1].
The main tables also contain comparison methods; those implementations, mechanism
analyses and ablations are outside this runner.

### Feature contract

```text
datasets/
  raw/<dataset>/                 # Original downloads or licensed local files
  processed/<collection>/       # Prepared images/audio/sensor inputs
  features/<backbone>/           # Reusable observation cache
  retrieval/<backbone>/<relation>.npz
  classification/<backbone>/<relation>.npz
```

| Array | Shape | Meaning |
|---|---|---|
| `a_train`, `h_a_train` | N × d, N × h | Paired A-H source features |
| `b_train`, `h_b_train` | N × d, N × h | Paired B-H source features |
| `a_test`, `b_test` | M × d | Retrieval candidates/queries in paired order |
| `positive_labels` | M, optional | Positive groups for category-positive retrieval |
| `prototypes` | C × d | A-side class prototypes constructed from training observations or registered text templates |
| `queries`, `labels` | M × d, M integers | Classification test queries and labels |

Each generated archive has a JSON sidecar identifying its protocol, encoder
runtime and row counts. Pickle objects are unsupported. Classification prototypes
are mean normalized training-A features, followed by normalization. VGGSound-309
uses the registered CLIP 80-template text ensemble. The official NYUv2 text hub
uses LanguageBind's depth-specific 80 templates. Test labels are used for scoring.

### HSA configuration

The fixed fitting settings follow the main implementation: trace-scaled ridge 0.5,
rank threshold multiplier 2, maximum rank 256, rank-null seed 142, 256 coordinate
permutations with seed 42 and quantile 0.95, and mismatch seeds 42–46.

```bash
python run.py --task retrieval --device cuda
python run.py --task classification --output outputs/classification.json
python run.py --dataset utd-mhad --batch-size 128
python -m unittest discover -s tests -v
```

### Project structure

```text
HSA/
  download.py             # Original data and frozen weights
  extract.py              # Shared encoders, raw-input adapters, task construction
  run.py                  # HSA fitting and evaluation
  hsa.py                  # Core relation recovery and score
  protocols.py            # Shared configuration reader and experiment selection
  configs/                # Sample IDs, splits, prompts and download versions
  environments/           # Pinned extraction dependencies
  tests/                  # Protocol, caching and download checks
  assets/                 # Manuscript overview and method figures
```

Encoder and data preparation provenance is recorded in [THIRD_PARTY.md](THIRD_PARTY.md).

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
