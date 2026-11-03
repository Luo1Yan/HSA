# Feature archives

Prepared features let you run HSA fitting and evaluation without loading the
backbone encoders. Install the evaluation dependencies and point the runner to
the directory containing your archives:

```bash
python -m pip install -r requirements.txt
python run.py --data-root path/to/features --check-data
python run.py --data-root path/to/features --device cpu --output outputs/features.json
```

Use `--dataset`, `--backbone`, `--task` or `--relation` to select a subset. The
runner uses the registered experiments in `configs/experiments.json`; `--list`
shows their identifiers. GPU retrieval scoring is available with `--device cuda`
and a compatible PyTorch installation. Classification fitting and scoring use
NumPy on the CPU.

## Directory layout

Place archives directly under the selected data root:

```text
path/to/features/
  retrieval/
    imagebind/depth_imu__utd.npz
    imagebind/depth_imu__utd.json
    languagebind/<relation>.npz
    languagebind/<relation>.json
  classification/
    imagebind/depth_imu__utd.npz
    imagebind/depth_imu__utd.json
    languagebind/<relation>.npz
    languagebind/<relation>.json
```

The full main evaluation uses 19 retrieval archives and 11 classification
archives. Raw data, encoder weights and intermediate feature caches are separate
inputs to the extraction workflow.

## Arrays

Each `.npz` is loaded with `allow_pickle=False`. Use finite numerical feature
matrices and preserve the sample ordering in the corresponding configuration.
Here, `N` is the training count, `M` the test count, `D` the target feature
dimension, `H` the hub feature dimension and `C` the number of classes.

| Array | Shape | Meaning |
|---|---|---|
| `a_train` | `(N, D)` | Training features for modality A |
| `h_a_train` | `(N, H)` | Hub features paired row by row with `a_train` |
| `b_train` | `(N, D)` | Training features for modality B |
| `h_b_train` | `(N, H)` | Hub features paired row by row with `b_train` |
| `a_test` | `(M, D)` | Retrieval queries/candidates in modality A |
| `b_test` | `(M, D)` | Retrieval queries/candidates in modality B |
| `positive_labels` | `(M,)` | Optional retrieval labels: equal labels identify positives |
| `prototypes` | `(C, D)` | Classification prototype bank in modality A |
| `queries` | `(M, D)` | Classification queries in modality B |
| `labels` | `(M,)` | Integer indices into the prototype bank |

Every archive contains the four training matrices. Retrieval additionally uses
`a_test` and `b_test`; with `positive_labels`, any candidate of the same label is
positive. Without these labels, the positive candidate has the same row index.
Classification additionally uses `prototypes`, `queries` and `labels`, with
labels between `0` and `C - 1`.

The published configurations use shared training sample IDs across both hub
edges, and `h_a_train` and `h_b_train` contain the same hub observations. Fitting
uses the A–hub and B–hub pairs. Classification prototypes are constructed from
training observations or the configured text ensemble; test labels determine
evaluation accuracy. The runner applies row normalization to classification
training features, prototypes and queries. Preserve the extraction protocol when
preparing retrieval features.

## Metadata and integrity

`extract.py` writes a companion `.json` with the same basename as each archive.
It records `experiment`, `protocol_sha256`, extraction version and source hash,
environment, and train/test counts. Keep this file with the archive. When present,
the runner checks `protocol_sha256` against the current configuration.

`--check-data` validates required arrays, finite values, compatible dimensions,
labels, row counts and any available protocol digest.
