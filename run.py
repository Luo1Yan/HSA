"""Run HSA and frozen-cosine main retrieval / prototype-classification experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hsa import HSA, l2_rows, topk_accuracy


RETRIEVAL = {
    "imagebind": (
        "text_audio__vggsound", "text_thermal__tartan_category", "text_imu__ego4d_gyro",
        "audio_depth__batvision", "audio_thermal__mavd", "audio_imu__ego4d_gyro",
        "depth_thermal__tartan", "depth_imu__utd", "thermal_imu__caltech_gyro",
    ),
    "languagebind": (
        "image-video", "image-thermal", "image-depth", "image-audio", "video-thermal",
        "video-depth", "video-audio", "thermal-depth", "thermal-audio", "depth-audio",
    ),
}
CLASSIFICATION = {
    "imagebind": (
        "text_audio__vggsound", "text_thermal__tartan_category", "text_imu__ego4d_gyro",
        "audio_depth__batvision", "audio_imu__ego4d_gyro", "depth_imu__utd",
    ),
    "languagebind": ("image-depth", "video-thermal", "video-depth", "video-audio", "thermal-depth"),
}


def inventory(task="all", backbone="all", relation=None):
    return [(name, model, edge)
            for name, registry in (("retrieval", RETRIEVAL), ("classification", CLASSIFICATION))
            if task in ("all", name)
            for model, edges in registry.items() if backbone in ("all", model)
            for edge in edges if relation is None or edge == relation]


def load_features(path, task):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    required = {"a_train", "h_a_train", "b_train", "h_b_train"}
    required |= {"a_test", "b_test"} if task == "retrieval" else {"prototypes", "queries", "labels"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path}: missing arrays {sorted(missing)}")
    for key in required - {"labels"}:
        array = data[key]
        if array.ndim != 2 or not len(array) or not np.isfinite(array).all():
            raise ValueError(f"{path}: {key} must be a nonempty finite matrix")
    a, ha, b, hb = (data[key] for key in ("a_train", "h_a_train", "b_train", "h_b_train"))
    if not len(a) == len(ha) == len(b) == len(hb) or len(a) < 2:
        raise ValueError(f"{path}: source rows must have matching counts >= 2")
    if a.shape[1] != b.shape[1] or ha.shape[1] != hb.shape[1]:
        raise ValueError(f"{path}: incompatible feature dimensions")
    if task == "retrieval":
        if data["a_test"].shape != data["b_test"].shape or data["a_test"].shape[1] != a.shape[1]:
            raise ValueError(f"{path}: retrieval test rows must be paired and dimensionally compatible")
        if "positive_labels" in data and data["positive_labels"].shape != (len(data["a_test"]),):
            raise ValueError(f"{path}: positive_labels must align with test rows")
    else:
        labels = data["labels"]
        if (labels.shape != (len(data["queries"]),) or not np.issubdtype(labels.dtype, np.integer)
                or labels.min() < 0 or labels.max() >= len(data["prototypes"])):
            raise ValueError(f"{path}: labels must index the common prototype bank")
        if any(data[key].shape[1] != a.shape[1] for key in ("prototypes", "queries")):
            raise ValueError(f"{path}: incompatible prototype/query dimensions")
    return data


def retrieval_metrics(data, model, device, batch_size):
    a, b = data["a_test"], data["b_test"]
    coordinates = model.retrieval_coordinates(a, b, device)
    raw = [torch.as_tensor(l2_rows(x), device=device) for x in (a, b)]
    labels = data.get("positive_labels")
    if labels is not None:
        labels = torch.as_tensor(np.unique(labels.astype(str), return_inverse=True)[1], device=device)
    result = {}
    with torch.no_grad():
        for method in ("FrozenCosine", "HSA"):
            directions = []
            for reverse in (False, True):
                counts = {k: 0 for k in (1, 5, 10)}
                for start in range(0, len(a), batch_size):
                    stop = min(start + batch_size, len(a))
                    scores = (model.retrieval_scores(coordinates, start, stop, reverse)
                              if method == "HSA" else raw[int(reverse)][start:stop] @ raw[1-int(reverse)].T)
                    order = torch.topk(scores, min(10, len(a)), dim=1).indices
                    matches = (order == torch.arange(start, stop, device=device)[:, None]
                               if labels is None else labels[order] == labels[start:stop, None])
                    for k in counts:
                        counts[k] += int(matches[:, :min(k, len(a))].any(dim=1).sum())
                directions.append({f"R@{k}": value / len(a) for k, value in counts.items()})
            result[method] = {key: (directions[0][key] + directions[1][key]) / 2 for key in directions[0]}
            result[method]["A_to_B"] = directions[0]
            result[method]["B_to_A"] = directions[1]
    return result


def evaluate(data, task, device, batch_size):
    # Classification source inputs follow the published normalized feature protocol.
    source = [data[key] for key in ("a_train", "h_a_train", "b_train", "h_b_train")]
    if task == "classification":
        source = [l2_rows(x) for x in source]
    model = HSA().fit(*source)
    if task == "retrieval":
        metrics = retrieval_metrics(data, model, device, batch_size)
    else:
        prototypes, queries = (l2_rows(data[key]) for key in ("prototypes", "queries"))
        scores = {"FrozenCosine": queries @ prototypes.T, "HSA": model.classify(prototypes, queries)}
        metrics = {name: topk_accuracy(values, data["labels"]) for name, values in scores.items()}
    return {"metrics": metrics, "fit": {
        "carrier_rank": model.model["hub_rank"], "resolution_rank": model.model["analytic_rank"],
        "gate": model.gate["gate"], "carrier_scale": model.calibration["hub_std"],
        "resolution_scale": model.calibration["residual_std"],
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output", type=Path, default=Path("outputs/main.json"))
    parser.add_argument("--task", choices=("all", "retrieval", "classification"), default="all")
    parser.add_argument("--backbone", choices=("all", "imagebind", "languagebind"), default="all")
    parser.add_argument("--relation", help="Evaluate one relation ID from --list")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--list", action="store_true", help="List selected main experiments")
    parser.add_argument("--check-data", action="store_true", help="Validate feature archives without fitting")
    args = parser.parse_args()
    selected = inventory(args.task, args.backbone, args.relation)
    if not selected or args.batch_size < 1:
        parser.error("select a registered main relation and a positive batch size")
    paths = [(task, backbone, relation, args.data_root / task / backbone / (relation + ".npz"))
             for task, backbone, relation in selected]
    if args.list:
        for task, backbone, relation, path in paths:
            print(f"{task}/{backbone}/{relation}")
        return
    missing = [str(path) for _, _, _, path in paths if not path.is_file()]
    if missing:
        parser.error("Missing prepared feature archives. See bash datasets.sh --help.\n" + "\n".join(missing))
    rows = []
    for task, backbone, relation, path in paths:
        data = load_features(path, task)
        print(f"{'Checked' if args.check_data else 'Evaluating'} {task}/{backbone}/{relation}", flush=True)
        if not args.check_data:
            row = {"task": task, "backbone": backbone, "relation": relation,
                   **evaluate(data, task, args.device, args.batch_size)}
            rows.append(row)
    if args.check_data:
        return
    aggregate = {}
    for task in ("retrieval", "classification"):
        group = [row for row in rows if row["task"] == task]
        if group:
            key = "R@10" if task == "retrieval" else "macro_top1"
            aggregate[task] = {"relations": len(group), "metric": key,
                               **{method: float(np.mean([row["metrics"][method][key] for row in group]))
                                  for method in ("FrozenCosine", "HSA")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"results": rows, "aggregate": aggregate}, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
