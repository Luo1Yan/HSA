"""Shared experiment definitions for acquisition, extraction and evaluation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CONFIG_ROOT = Path(__file__).resolve().parent / "configs"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def experiments(dataset="all", backbone="all", task="all", relation=None):
    rows = read_json(CONFIG_ROOT / "experiments.json")["experiments"]
    return [row for row in rows
            if dataset in ("all", row["dataset"])
            and backbone in ("all", row["backbone"])
            and task in ("all", row["task"])
            and (relation is None or relation == row["relation"])]


def dataset_names():
    return sorted({row["dataset"] for row in experiments()})


def selection_arguments(parser):
    parser.add_argument("--dataset", choices=["all", *dataset_names()], default="all")
    parser.add_argument("--backbone", choices=("all", "imagebind", "languagebind"), default="all")
    parser.add_argument("--task", choices=("all", "retrieval", "classification"), default="all")
    parser.add_argument("--relation", help="One relation ID from --list")
    parser.add_argument("--data-root", type=Path, default=Path("datasets"))
    parser.add_argument("--list", action="store_true", help="List matching experiments without running")


def select(args):
    rows = experiments(args.dataset, args.backbone, args.task, args.relation)
    if not rows:
        raise ValueError("No experiment matches the selected dataset/backbone/task/relation")
    return rows


def experiment_id(row):
    return "/".join(row[key] for key in ("task", "backbone", "relation"))


def protocol_digest(row):
    """Include observations and split order; extraction caches cannot cross protocols."""
    description = dict(experiment=row,
                       records=read_json(CONFIG_ROOT / (row["collection"] + ".json")),
                       templates=read_json(CONFIG_ROOT / "text_templates.json"),
                       models=read_json(CONFIG_ROOT / "downloads.json")["models"][row["backbone"]])
    # JSON formatting and platform line endings do not change a scientific protocol.
    return hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()


def safe_path(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Path escapes its data directory: {relative}")
    return target
