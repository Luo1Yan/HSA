"""Extract frozen representations and construct the registered HSA task inputs.

One shared encoder, observation cache and task builder serve all datasets.
Dataset adapters retain the preprocessing used by the corresponding experiment.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import random
import re
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from protocols import (CONFIG_ROOT, experiment_id, protocol_digest, read_json,
                       safe_path, select, selection_arguments)

EXTRACTION_VERSION = 1
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


def normalize(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def assemble(recipe, features, text_prototypes=None):
    """Construct paired hub edges and readouts without consulting test labels in fit."""
    def rows(ids, modality):
        return normalize(np.stack([features[(str(key), modality)] for key in ids]))
    train, test = recipe["train_ids"], recipe["test_ids"]
    a, h, b = (rows(train, recipe[key]) for key in ("a", "hub", "b"))
    result = dict(a_train=a, h_a_train=h, b_train=b, h_b_train=h.copy())
    if recipe["task"] == "retrieval":
        result.update(a_test=rows(test, recipe["a"]), b_test=rows(test, recipe["b"]))
        if "positive_labels" in recipe:
            result["positive_labels"] = np.asarray(recipe["positive_labels"])
    else:
        labels = np.asarray(recipe["train_labels"], dtype=np.int64)
        count = len(recipe["class_names"])
        if not np.array_equal(np.unique(labels), np.arange(count)):
            raise ValueError("Each prototype class must have training observations")
        prototypes = (normalize(text_prototypes) if recipe.get("prototype") == "text-ensemble"
                      else normalize(np.stack([a[labels == k].mean(axis=0) for k in range(count)])))
        result.update(prototypes=prototypes, queries=rows(test, recipe["b"]),
                      labels=np.asarray(recipe["test_labels"], dtype=np.int64))
    return result


def required_observations(recipe):
    required = {}
    for key in recipe["train_ids"]:
        required.setdefault(key, set()).update([recipe["a"], recipe["hub"], recipe["b"]])
    for key in recipe["test_ids"]:
        required.setdefault(key, set()).update(
            [recipe["a"], recipe["b"]] if recipe["task"] == "retrieval" else [recipe["b"]])
    return required


class FrozenEncoder:
    """Identical frozen-model API for every dataset; one modality resides on GPU."""

    def __init__(self, backbone, model_root, device, profile="standard", msrvtt=False):
        import torch
        self.torch = torch
        self.backbone, self.device, self.profile = backbone, device, profile
        self.root = Path(model_root) / backbone
        self.msrvtt = msrvtt
        self.current = None
        self.model = None
        source = self.root / "source"
        if not source.is_dir():
            raise FileNotFoundError(f"Missing encoder source: {source}. Run download.py --models-only")
        sys.path.insert(0, str(source.resolve()))
        encoder_compatibility()
        if backbone == "imagebind":
            from imagebind import data
            from imagebind.models.imagebind_model import imagebind_huge
            self.data = data
            self.model = imagebind_huge(pretrained=False)
            state = torch.load(self.root / "checkpoints/imagebind_huge.pth", map_location="cpu", weights_only=True)
            self.model.load_state_dict(state, strict=True)
            self.model.eval().requires_grad_(False).to(device)
        else:
            import transformers
            import peft
            expected = ("4.57.1", "0.17.1") if profile == "classification" else ("4.30.2", "0.4.0.dev0")
            if (transformers.__version__, peft.__version__) != expected:
                raise RuntimeError(f"This protocol requires transformers={expected[0]}, peft={expected[1]}. "
                                   f"Install environments/{profile}.txt in its own environment.")
            from languagebind import model_dict, config_dict, transform_dict, LanguageBindImageTokenizer
            self.classes, self.configs, self.transforms = model_dict, config_dict, transform_dict
            self.tokenizer = LanguageBindImageTokenizer.from_pretrained(
                str(self.root / "checkpoints/Image"), local_files_only=True)

    def close(self):
        self.model = None
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def load_modality(self, modality):
        if self.backbone == "imagebind":
            return
        key = ("audio" if self.msrvtt else "depth") if modality == "text" else modality
        if self.current == key:
            return
        self.close()
        names = dict(image="Image", depth="Depth", thermal="Thermal",
                     video="Video" if self.msrvtt else "Video_FT",
                     audio="Audio" if self.msrvtt else "Audio_FT")
        directory = self.root / "checkpoints" / names[key]
        if self.profile == "classification":
            config = self.configs[key].from_pretrained(str(directory), local_files_only=True)
            self.model = self.classes[key](config)
            state = self.torch.load(directory / "pytorch_model.bin", map_location="cpu", weights_only=True)
            targets = self.model.state_dict()
            converted = {}
            for name, value in state.items():
                candidate = name.rsplit(".", 1)
                changed = candidate[0] + ".base_layer." + candidate[1] if len(candidate) == 2 else name
                converted[changed if name not in targets and changed in targets else name] = value
            missing, unexpected = self.model.load_state_dict(converted, strict=False)
            if any(not name.endswith("position_ids") for name in missing + unexpected):
                raise RuntimeError(f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}")
        else:
            self.model = self.classes[key].from_pretrained(str(directory), local_files_only=True)
        for module in self.model.modules():
            if hasattr(module, "config"):
                module.config._attn_implementation = "eager"
        self.model.eval().requires_grad_(False).to(self.device)
        self.transform = self.transforms[key](self.model.config)
        self.current = key

    def encode(self, values, modality, tensor=False, sample_ids=None):
        torch = self.torch
        self.load_modality(modality)
        with torch.inference_mode():
            if self.backbone == "imagebind":
                key = "vision" if modality in ("image", "video") else modality
                if tensor:
                    batch = torch.stack(values).to(self.device)
                elif modality in ("image", "audio", "text"):
                    loader = {"image": self.data.load_and_transform_vision_data,
                              "audio": self.data.load_and_transform_audio_data,
                              "text": self.data.load_and_transform_text}[modality]
                    batch = loader(values, self.device)
                elif modality in ("depth", "thermal"):
                    batch = torch.stack([imagebind_scalar_image(p, modality) for p in values]).to(self.device)
                else:
                    raise ValueError(f"Expected prepared tensors for {modality}")
                output = self.model({key: batch})[key]
            elif modality == "text":
                batch = self.tokenizer(values, max_length=77, padding="max_length", truncation=True, return_tensors="pt")
                output = self.model.get_text_features(**{k: v.to(self.device) for k, v in batch.items()})
            else:
                if tensor:
                    batch = {"pixel_values": torch.stack(values)}
                elif sample_ids is not None:
                    # Upstream video preprocessing includes random horizontal flips.
                    # Fix each observation independently of task, batch and resume order.
                    tensors = []
                    for value, sample_id in zip(values, sample_ids):
                        seed = int(hashlib.sha256(str(sample_id).encode()).hexdigest()[:8], 16)
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        tensors.append(self.transform([value])["pixel_values"][0])
                    batch = {"pixel_values": torch.stack(tensors)}
                else:
                    batch = self.transform(values)
                output = self.model.get_image_features(**{k: v.to(self.device) for k, v in batch.items()})
            return torch.nn.functional.normalize(output.float(), dim=-1).cpu().numpy()


def imagebind_scalar_image(path, modality):
    from PIL import Image
    from torchvision import transforms as T
    image = Image.open(path).convert("L")
    if modality == "depth":
        image = Image.fromarray(np.asarray(image, dtype=np.float32) / 255.0)
    ops = [T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
    if modality == "thermal":
        ops.append(T.Normalize((0.5,), (0.5,)))
    return T.Compose(ops)(image)


def encoder_compatibility():
    """Keep the pinned upstream encoders importable in both recorded runtimes."""
    import torch
    try:
        import torchvision.transforms.functional_tensor
    except ModuleNotFoundError:
        from torchvision.transforms import _functional_tensor
        sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor
    from transformers.models.clip import modeling_clip
    if not hasattr(modeling_clip, "_expand_mask"):
        def expand_mask(mask, dtype, tgt_len=None):
            batch, source = mask.size()
            target = tgt_len if tgt_len is not None else source
            expanded = mask[:, None, None, :].expand(batch, 1, target, source).to(dtype)
            inverted = 1.0 - expanded
            return inverted.masked_fill(inverted.to(torch.bool), torch.finfo(dtype).min)
        modeling_clip._expand_mask = expand_mask


def depth_image(array):
    """Registered inverse-depth quantization for TartanRGBT and BatVision."""
    array = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(array) & (array > 0)
    if not valid.any():
        return np.zeros(array.shape, dtype=np.uint8)
    low, high = np.percentile(array[valid], [1, 99]).astype(np.float32)
    inverse = 1.0 / np.maximum(np.clip(array, low, high), 1e-6)
    inverse[~valid] = 0
    low, high = np.percentile(inverse[valid], [1, 99]).astype(np.float32)
    return (np.clip((inverse - low) / (high - low), 0, 1) * 255).astype(np.uint8) if high > low else np.zeros(array.shape, dtype=np.uint8)


def run_ffmpeg(arguments):
    executable = shutil.which("ffmpeg")
    if executable is None:
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError("Install imageio-ffmpeg or put FFmpeg on PATH for audio/video decoding") from exc
    subprocess.run([executable, "-hide_banner", "-loglevel", "error", "-y", *map(str, arguments)], check=True)


def middle_frame(path, output, quality=95, opencv=False):
    from PIL import Image
    if output.is_file():
        return str(output)
    if opencv:
        import cv2
        cap = cv2.VideoCapture(str(path))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok or not cv2.imwrite(str(output), frame):
                raise RuntimeError(f"Cannot decode RGB frame: {path}")
        finally:
            cap.release()
    else:
        import decord
        decord.bridge.set_bridge("native")
        reader = decord.VideoReader(str(path), num_threads=1)
        Image.fromarray(reader[len(reader) // 2].asnumpy()).save(output, quality=quality)
    return str(output)


def vgg_media(path, frame, audio):
    if frame.is_file() and audio.is_file():
        return
    try:
        import decord
        import soundfile as sf
        middle_frame(path, frame)
        reader = decord.AudioReader(str(path), sample_rate=16000, mono=True)
        waveform = np.asarray(reader[:].asnumpy()).squeeze()
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=0)
        if not waveform.size:
            raise ValueError("Empty audio stream")
        sf.write(audio, waveform.astype(np.float32), 16000)
    except Exception:
        run_ffmpeg(["-ss", "5", "-i", path, "-frames:v", "1", "-q:v", "2", frame])
        run_ffmpeg(["-i", path, "-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio])


def action_tensor(path, modality):
    import torch
    from scipy.io import loadmat
    from torch.nn import functional as F
    if modality == "depth":
        value = loadmat(path)["d_depth"].astype(np.float32)
        value = value[:, :, value.shape[2] // 2]
        valid = value > 0
        low, high = np.percentile(value[valid], [1, 99])
        value = np.clip((value - low) / max(high - low, 1e-6), 0, 1)
        return F.interpolate(torch.from_numpy(value)[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0]
    value = loadmat(path)["d_iner"].astype(np.float32).T
    value = (value - value.mean(axis=1, keepdims=True)) / np.maximum(value.std(axis=1, keepdims=True), 1e-6)
    return F.interpolate(torch.from_numpy(value)[None], size=2000, mode="linear", align_corners=False)[0]


def full_audio_tensor(path):
    import torch
    import torchaudio
    import soundfile as sf
    from imagebind import data
    wave, rate = sf.read(path, dtype="float32", always_2d=True)
    wave = torch.from_numpy(wave.mean(axis=1)).unsqueeze(0)
    if rate != 16000:
        wave = torchaudio.functional.resample(wave, rate, 16000)
    mel = data.waveform2melspec(wave, 16000, 128, 204)
    return ((mel + 4.268) / 9.138).unsqueeze(0)


class RawInputs:
    """Resolve a registered observation to model inputs beneath one data root."""

    def __init__(self, data_root, collection, records):
        self.collection = collection
        self.dataset = read_json(CONFIG_ROOT / (collection + ".json"))["dataset"]
        self.root = Path(data_root) / "raw" / self.dataset
        self.work = Path(data_root) / "processed" / collection
        self.work.mkdir(parents=True, exist_ok=True)
        self.records = {r["id"]: r for r in records}
        self.file_index = None
        self.auxiliary = {}

    def find(self, filename, contains=None):
        if self.file_index is None:
            self.file_index = {}
            for path in self.root.rglob("*"):
                if path.is_file():
                    self.file_index.setdefault(path.name, []).append(path)
        found = self.file_index.get(filename, [])
        if contains:
            found = [p for p in found if contains in p.as_posix()]
        if len(found) != 1:
            raise FileNotFoundError(f"Expected one {filename} under {self.root}"
                                    f"{(' containing ' + contains) if contains else ''}; found {len(found)}")
        return found[0]

    def workfile(self, row, suffix):
        key = hashlib.sha256(row["id"].encode()).hexdigest()[:24]
        return self.work / (key + suffix)

    def nyu_shards(self, split):
        # The official mirror has also used two validation shards. Preserve their
        # ordered contents rather than assuming a particular tar partition size.
        cache_key = "nyu_shards_" + split
        if cache_key in self.auxiliary:
            return self.auxiliary[cache_key]
        files = sorted(self.root.rglob(f"{split}-*.tar"), key=lambda p: p.name)
        names = [p.name for p in files]
        expected = [f"{split}-{i:06d}.tar" for i in range(len(files))]
        if not files or names != expected:
            raise FileNotFoundError(f"Missing or duplicated ordered NYUv2 {split} shards below {self.root}")
        self.auxiliary[cache_key] = files
        return files

    def check(self, row, modality):
        """Check original files without loading models or decoding media."""
        return self.input(row, modality, backbone=None, check=True)

    def input(self, row, modality, backbone, check=False):
        from PIL import Image
        col, sid = self.collection, row["id"]
        if modality == "text" and col != "ego4d":
            return row.get("text", ""), False
        if col == "utd-mhad":
            suffix = {"image": "_color.avi", "depth": "_depth.mat", "imu": "_inertial.mat"}[modality]
            path = self.find(sid + suffix)
            if check:
                return path
            if modality == "image":
                return middle_frame(path, self.workfile(row, ".jpg"), opencv=True), False
            return action_tensor(path, modality), True
        if col.startswith("vggsound"):
            if row.get("split") == "test":
                path = self.find(row["parquet"])
                if check:
                    return path
                import pyarrow.parquet as pq
                cache_key = (str(path), row["row_group"])
                if self.auxiliary.get("parquet_key") != cache_key:
                    self.auxiliary["parquet_audio"] = pq.ParquetFile(path).read_row_group(row["row_group"], columns=["audio"]).column("audio").to_pylist()
                    self.auxiliary["parquet_key"] = cache_key
                audio = self.auxiliary["parquet_audio"][row["row"]]
                if Path(audio["path"]).stem != Path(sid).stem:
                    raise ValueError(f"Parquet row identity changed: {sid}")
                output = self.workfile(row, ".wav")
                if not output.is_file():
                    output.write_bytes(audio["bytes"])
                return str(output), False
            path = self.find(sid if sid.endswith(".mp4") else sid + ".mp4")
            if check:
                return path
            frame, audio = self.workfile(row, ".jpg"), self.workfile(row, ".wav")
            vgg_media(path, frame, audio)
            return str(frame if modality == "image" else audio), False
        if col == "ucf101":
            path = self.find(Path(row["video"]).name)
            if check:
                return path
            return (middle_frame(path, self.workfile(row, ".jpg"), quality=92) if modality == "image" else str(path)), False
        if col == "msrvtt":
            path = self.find(sid + ".mp4")
            if check:
                return path
            if modality == "audio":
                output = self.workfile(row, ".wav")
                if not output.is_file():
                    status = extract_wav_with_pyav(path, output)
                    if not status["ok"]:
                        raise RuntimeError(f"Cannot decode {path}: {status}")
                path = output
            return str(path), False
        if col in ("tartan-frames", "tartan-clips"):
            trajectory, frame = sid.rsplit("/", 1)
            trajectory = trajectory.split("/")[-1]
            def framepath(fid, kind):
                name = fid + ("_rgb_in_thermal.png" if kind == "image" else ".png" if kind == "thermal" else ".npy")
                return self.find(name, trajectory)
            if modality == "video":
                frame_ids = row["frames"]
                paths = [framepath(fid, "image") for fid in frame_ids]
                if check:
                    return paths
                output = self.workfile(row, ".mp4")
                if not output.is_file():
                    write_video(output, paths)
                return str(output), False
            path = framepath(frame, modality)
            if check:
                return path
            if modality == "depth":
                output = self.workfile(row, "_depth.png")
                if not output.is_file():
                    Image.fromarray(depth_image(np.load(path))).save(output)
                path = output
            if modality == "thermal" and backbone == "languagebind":
                output = self.workfile(row, "_thermal_rgb.png")
                if not output.is_file():
                    Image.open(path).convert("RGB").save(output)
                path = output
            return str(path), False
        if col == "nyuv2-cmcl":
            split = "train" if row["split"] == "train" else "val"
            folder = self.work / split / row["folder"]
            if check:
                return self.nyu_shards(split)
            if not all((folder / name).is_file() for name in ("rgb.png", "depth.png")):
                self.prepare_nyu(split)
            return str(folder / ("rgb.png" if modality == "image" else "depth.png")), False
        if col == "nyuv2-official":
            path = self.find("nyu_depth_v2_labeled.mat")
            if check:
                return path
            import h5py
            import torch
            from torchvision import transforms as T
            with h5py.File(path, "r") as f:
                index = int(sid) - 1
                if modality == "image":
                    array = np.asarray(f["images"][index], dtype=np.uint8).transpose(2, 1, 0)
                    tensor = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGE_MEAN, IMAGE_STD)])(Image.fromarray(array))
                else:
                    array = np.asarray(f["depths"][index], dtype=np.float32).T * 1000
                    transform = self.auxiliary.get("metric_depth_transform")
                    if transform is None:
                        raise RuntimeError("Depth transform was not initialized")
                    tensor = transform(array)
            return tensor, True
        if col == "batvision":
            if "bat_rows" not in self.auxiliary:
                mapping = {}
                for path in sorted(self.root.rglob("*.csv")):
                    if path.name not in ("train.csv", "val.csv", "test.csv"):
                        continue
                    with path.open(encoding="utf-8-sig", newline="") as f:
                        for item in csv.DictReader(f):
                            mapping.setdefault(path.parent.name + "/" + item["index"], item)
                self.auxiliary["bat_rows"] = mapping
            item = self.auxiliary["bat_rows"][sid]
            key = "camera" if modality == "image" else modality
            path = self.find(item[key + " file name"], sid.split('/')[0])
            if check:
                return path
            if modality == "depth":
                output = self.workfile(row, "_depth.png")
                if not output.is_file():
                    Image.fromarray(depth_image(np.load(path))).save(output)
                path = output
            if modality == "audio" and backbone == "imagebind":
                return full_audio_tensor(path), True
            return str(path), False
        if col == "mavd":
            drive, key = sid.split("/", 1)
            names = {"image": f"fl_rgb_{key}.jpg", "thermal": f"fl_ir_aligned_{key}.jpg", "audio": f"audio_0_{key}.mp3"}
            folder = {"image": "fl_rgb", "thermal": "fl_ir_aligned", "audio": "audio"}[modality]
            path = self.find(names[modality], drive + "/" + folder + "/")
            if check:
                return path
            if modality == "audio":
                output = self.workfile(row, ".wav")
                if not output.is_file():
                    run_ffmpeg(["-i", path, "-vn", "-ac", "1", "-ar", "16000", output])
                if backbone == "imagebind":
                    return full_audio_tensor(output), True
                path = output
            if modality == "thermal" and backbone == "languagebind":
                output = self.workfile(row, "_thermal_rgb.png")
                if not output.is_file():
                    Image.open(path).convert("RGB").save(output)
                path = output
            return str(path), False
        if col == "ego4d":
            if modality == "text":
                path = self.find("narration.json")
                if check:
                    return path
                if "narration" not in self.auxiliary:
                    self.auxiliary["narration"] = read_json(path)
                ts, text, _, _ = select_narration(sid, self.auxiliary["narration"], row["duration"])
                if abs(ts - row["timestamp"]) > 1e-5:
                    raise ValueError(f"Narration selection changed for {sid}")
                return text, False
            path = self.find(row["imu_file"] if modality == "imu" else sid + ".mp4")
            if check:
                return path
            if modality == "imu":
                import torch
                return torch.from_numpy(load_imu_window(str(path), row["timestamp"], "gyro_first")), True
            if modality == "video":
                return load_local_video_tensor(str(path), row["timestamp"], "cpu"), True
            return load_audio_tensor(str(path), row["timestamp"], "cpu"), True
        if col == "caltech-rgbt":
            bag = self.find("2022-01-08-15-21-59.bag")
            if check:
                return bag
            if "caltech" not in self.auxiliary:
                self.auxiliary["caltech"] = prepare_caltech(bag, self.work, list(self.records.values()))
            ts, values = self.auxiliary["caltech"]
            if modality == "imu":
                import torch
                array = imu_window_from_arrays(ts, values, row["timestamp"], "gyro_first", 2000, row["half_window"])
                return torch.from_numpy(array), True
            return str(self.work / (sid + (".jpg" if modality == "image" else ".png"))), False
        raise ValueError(f"Unknown observation collection: {col}")

    def prepare_nyu(self, split):
        """Preserve the CMCL archive order and per-image depth quantization."""
        import h5py
        from PIL import Image
        index = 0
        for shard in self.nyu_shards(split):
            with tarfile.open(shard) as archive:
                for member in archive:
                    if not member.name.endswith(".h5"):
                        continue
                    label = member.name.split("/")[1].split("_")[0]
                    target = self.work / split / f"{label}_{index}" / "0"
                    target.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as stream, h5py.File(io.BytesIO(stream.read()), "r") as data:
                        rgb = np.asarray(data["rgb"]).transpose(1, 2, 0).astype(np.uint8)
                        depth = np.asarray(data["depth"])
                        depth = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(np.uint8)
                    Image.fromarray(rgb).save(target / "rgb.png")
                    Image.fromarray(depth).save(target / "depth.png")
                    index += 1
        expected = sum(r["split"] == ("train" if split == "train" else "test") for r in self.records.values())
        if index != expected:
            raise ValueError(f"NYUv2 {split}: expected {expected} observations, decoded {index}")


class FeatureStore:
    """One reusable observation cache shared by retrieval and classification."""

    def __init__(self, directory, recipe, batch_size):
        description = dict(backbone=recipe["backbone"], collection=recipe["collection"],
                           environment=recipe.get("environment", "standard"),
                           models=read_json(CONFIG_ROOT / "downloads.json")["models"][recipe["backbone"]],
                           extractor=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                           templates=read_json(CONFIG_ROOT / "text_templates.json"),
                           records=read_json(CONFIG_ROOT / (recipe["collection"] + ".json")))
        self.digest = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()
        directory = Path(directory) / recipe["backbone"] / recipe["collection"]
        directory.mkdir(parents=True, exist_ok=True)
        self.database = sqlite3.connect(directory / (self.digest[:20] + ".sqlite3"))
        self.database.execute("CREATE TABLE IF NOT EXISTS features (id TEXT, modality TEXT, dimension INTEGER, value BLOB, PRIMARY KEY(id,modality))")

    def get(self, key, modality):
        row = self.database.execute("SELECT dimension,value FROM features WHERE id=? AND modality=?", (key, modality)).fetchone()
        if row is None:
            return None
        value = np.frombuffer(row[1], dtype=np.float32).copy()
        if value.shape != (row[0],) or not np.isfinite(value).all():
            raise ValueError(f"Invalid cached feature: {key}/{modality}")
        return value

    def put(self, ids, modality, values):
        self.database.executemany("INSERT OR REPLACE INTO features VALUES (?,?,?,?)",
                                  [(key, modality, len(value), np.asarray(value, dtype=np.float32).tobytes()) for key, value in zip(ids, values)])
        self.database.commit()

    def close(self):
        self.database.close()


def extract_recipe(recipe, args):
    import torch
    digest = protocol_digest(recipe)
    output = args.data_root / (experiment_id(recipe) + ".npz")
    metadata_path = output.with_suffix(".json")
    if output.is_file() and metadata_path.is_file() and not args.force and not args.check_inputs:
        metadata = read_json(metadata_path)
        if (metadata.get("protocol_sha256") == digest and metadata.get("extraction_version") == EXTRACTION_VERSION
                and metadata.get("extractor_sha256") == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            from run import load_features
            load_features(output, recipe["task"])
            print(f"Already complete: {experiment_id(recipe)}", flush=True)
            return
        print(f"Rebuilding features for updated extraction code: {experiment_id(recipe)}", flush=True)
    records = read_json(CONFIG_ROOT / (recipe["collection"] + ".json"))["records"]
    raw = RawInputs(args.data_root, recipe["collection"], records)
    required = required_observations(recipe)
    missing = []
    for key, modalities in required.items():
        for modality in sorted(modalities):
            try:
                raw.check(raw.records[key], modality)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                missing.append({"id": key, "modality": modality, "error": str(exc)})
    if missing:
        report = args.data_root / "checks" / (recipe["collection"] + "_missing.json")
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(missing, indent=2), encoding="utf-8")
        raise FileNotFoundError(f"{len(missing)} missing inputs; see {report}. No samples were replaced.")
    if args.check_inputs:
        print(f"Inputs present: {experiment_id(recipe)} ({len(required)} observations)", flush=True)
        return
    profile = recipe.get("environment", "standard")
    encoder = FrozenEncoder(recipe["backbone"], args.model_root, args.device, profile, recipe["collection"] == "msrvtt")
    features = {}
    text_bank = None
    cache = FeatureStore(args.data_root / "features", recipe, args.batch_size)
    try:
        if recipe["collection"] in ("nyuv2-official", "vggsound-309"):
            templates = read_json(CONFIG_ROOT / "text_templates.json")[recipe["text_templates"]]
            prompts = [t.format(name) for name in recipe["class_names"] for t in templates]
            encoded = np.concatenate([encoder.encode(prompts[i:i + args.batch_size], "text") for i in range(0, len(prompts), args.batch_size)])
            text_bank = normalize(encoded.reshape(len(recipe["class_names"]), len(templates), -1).mean(axis=1))
        for modality in sorted({m for values in required.values() for m in values}):
            keys = [key for key, values in required.items() if modality in values]
            if modality == "text" and text_bank is not None:
                for key in keys:
                    features[(key, modality)] = text_bank[raw.records[key]["label"]]
                continue
            pending = []
            for key in keys:
                value = None if args.force else cache.get(key, modality)
                if value is None:
                    pending.append(key)
                else:
                    features[(key, modality)] = value
            if not pending:
                print(f"Reused {len(keys)} {modality} observations", flush=True)
                continue
            encoder.load_modality(modality)
            if recipe["collection"] == "nyuv2-official" and modality == "depth":
                from languagebind.depth.processing_depth import get_depth_transform
                raw.auxiliary["metric_depth_transform"] = get_depth_transform(encoder.model.config)
            for start in range(0, len(pending), args.batch_size):
                batch_keys = pending[start:start + args.batch_size]
                inputs = [raw.input(raw.records[key], modality, recipe["backbone"]) for key in batch_keys]
                if len({v[1] for v in inputs}) != 1:
                    raise ValueError("Mixed tensor/file input in one modality batch")
                values = encoder.encode([v[0] for v in inputs], modality, tensor=inputs[0][1], sample_ids=batch_keys)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite features: {modality} {batch_keys}")
                cache.put(batch_keys, modality, values)
                features.update({(key, modality): value for key, value in zip(batch_keys, values)})
                if start == 0 or start // args.batch_size % 50 == 0:
                    print(f"{recipe['collection']} {modality}: {start + len(batch_keys)}/{len(pending)} new observations", flush=True)
        arrays = assemble(recipe, features, text_bank)
        atomic_npz(output, **arrays)
        metadata = dict(experiment=experiment_id(recipe), protocol_sha256=digest,
                        extraction_version=EXTRACTION_VERSION, batch_size=args.batch_size,
                        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        environment=profile, torch=torch.__version__,
                        train_count=len(recipe["train_ids"]), test_count=len(recipe["test_ids"]))
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved {output}", flush=True)
    finally:
        cache.close()
        encoder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection_arguments(parser)
    parser.add_argument("--model-root", type=Path, default=Path("models"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-inputs", action="store_true", help="Check all registered original files without encoding")
    parser.add_argument("--force", action="store_true", help="Regenerate selected local feature outputs")
    parser.add_argument("--environment", choices=("standard", "classification"),
                        help="Select protocols for one of the two pinned extraction environments")
    args = parser.parse_args()
    try:
        recipes = select(args)
        if args.environment:
            recipes = [r for r in recipes if r.get("environment", "standard") == args.environment]
            if not recipes:
                raise ValueError("No matching protocol in this extraction environment")
        if args.batch_size < 1:
            raise ValueError("batch size must be positive")
        if args.list:
            for recipe in recipes:
                print(f"{experiment_id(recipe)}  dataset={recipe['dataset']}  collection={recipe['collection']}  environment={recipe.get('environment', 'standard')}")
            return
        initialize_runtime()
        for recipe in recipes:
            extract_recipe(recipe, args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")


# Dataset-specific temporal and sensor transforms follow below. All feed the same
# FrozenEncoder.encode interface and the same assemble function above.

def initialize_runtime():
    global torch, F, cv2, pd, av, AudioResampler, wave, Image
    import torch
    import torch.nn.functional as F
    import cv2
    import pandas as pd
    import av
    from av.audio.resampler import AudioResampler
    import wave
    from PIL import Image


def prepare_caltech(bag, work, records):
    from rosbags.rosbag1 import Reader
    from rosbags.typesys import Stores, get_typestore
    typestore = get_typestore(Stores.ROS1_NOETIC)
    remaining = sorted(records, key=lambda row: row["timestamp"])
    cursor = 0
    times, values = [], []
    rgb, rgb_time = None, None
    topics = {"/boson/thermal/image_raw", "/eo/color/image_color/compressed", "/imu/imu"}
    with Reader(bag) as reader:
        connections = [c for c in reader.connections if c.topic in topics]
        if {c.topic for c in connections} != topics:
            raise ValueError("Caltech bag is missing a required image or IMU topic")
        for connection, stamp, raw in reader.messages(connections=connections):
            timestamp = float(stamp) * 1e-9
            if timestamp > remaining[-1]["timestamp"] + 5.1:
                break
            message = typestore.deserialize_ros1(raw, connection.msgtype)
            if connection.topic == "/imu/imu":
                g, a = message.angular_velocity, message.linear_acceleration
                times.append(timestamp)
                values.append([g.x, g.y, g.z, a.x, a.y, a.z])
            elif connection.topic == "/eo/color/image_color/compressed":
                rgb = Image.open(io.BytesIO(bytes(message.data))).convert("RGB")
                rgb_time = timestamp
            elif cursor < len(remaining) and abs(timestamp - remaining[cursor]["timestamp"]) < 1e-6:
                row = remaining[cursor]
                if rgb is None or abs(timestamp - rgb_time) > 0.2:
                    raise ValueError(f"Unpaired Caltech image: {row['id']}")
                rgb.save(work / (row["id"] + ".jpg"), quality=95)
                image_msg_to_pil(message, False).save(work / (row["id"] + ".png"))
                cursor += 1
    if cursor != len(remaining):
        raise ValueError(f"Caltech bag contains {cursor}/{len(remaining)} registered image pairs")
    return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float32)


def clean_narration_text(text: str) -> str:
    text = str(text or "").strip()
    replacements = {
        "#C C": "The camera wearer",
        "#c c": "The camera wearer",
        "#C": "The camera wearer",
        "#c": "The camera wearer",
        " C ": " the camera wearer ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = " ".join(text.split())
    if text:
        text = text[0].upper() + text[1:]
    return text


def iter_narrations(narr_obj: Dict[str, Any]) -> Iterable[Tuple[str, float, str]]:
    for pass_name, pass_obj in narr_obj.items():
        if not isinstance(pass_obj, dict):
            continue
        for item in pass_obj.get("narrations", []):
            try:
                ts = float(item.get("timestamp_sec", item.get("_unmapped_timestamp_sec")))
            except Exception:
                continue
            txt = clean_narration_text(str(item.get("narration_text", "")))
            if not txt:
                continue
            yield str(pass_name), ts, txt


def select_narration(uid: str, narrations: Dict[str, Any], duration: float) -> Tuple[float, str, str, str]:
    candidates = []
    for pass_name, ts, txt in iter_narrations(narrations.get(uid, {})):
        if not np.isfinite(ts):
            continue
        ts = min(max(float(ts), 0.0), max(float(duration), 0.0))
        lower = txt.lower()
        penalty = 1000 if "#unsure" in lower or "unsure" in lower else 0
        candidates.append((penalty, -len(txt), ts, txt, pass_name))
    if not candidates:
        return 0.0, "The camera wearer performs an activity.", "missing", "fallback_no_narration"
    candidates.sort()
    _, _, ts, txt, pass_name = candidates[0]
    return float(ts), txt, pass_name, "longest_clean_narration_valid_timestamp"


def resize_short_side(video: torch.Tensor, size: int = 224) -> torch.Tensor:
    c, t, h, w = video.shape
    if h <= 0 or w <= 0:
        raise ValueError("Invalid frame size")
    if h < w:
        new_h = size
        new_w = int(round(w * size / h))
    else:
        new_w = size
        new_h = int(round(h * size / w))
    x = video.permute(1, 0, 2, 3)
    x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
    return x.permute(1, 0, 2, 3)


def spatial_crops(video: torch.Tensor, size: int = 224) -> List[torch.Tensor]:
    _, _, h, w = video.shape
    crops = []
    if h > w:
        offsets = [(0, 0), ((h - size) // 2, 0), (h - size, 0)]
    else:
        offsets = [(0, 0), (0, (w - size) // 2), (0, w - size)]
    for y, x in offsets:
        crops.append(video[:, :, y : y + size, x : x + size])
    return crops


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=video.dtype)[:, None, None, None]
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=video.dtype)[:, None, None, None]
    return (video - mean) / std


def read_frame(cap: cv2.VideoCapture, frame_idx: int, fallback: np.ndarray | None = None) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
    ok, frame = cap.read()
    if not ok or frame is None:
        if fallback is not None:
            return fallback
        raise RuntimeError(f"Could not read frame {frame_idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_local_video_tensor(video_path: str, timestamp_sec: float, device: torch.device) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 and frame_count > 0 else max(timestamp_sec + 1.0, 2.0)
    centers = np.linspace(timestamp_sec - 2.0, timestamp_sec + 2.0, 5)
    centers = np.clip(centers, 0.0, max(0.0, duration - 0.01))
    clips: List[torch.Tensor] = []
    fallback = None
    try:
        for center in centers:
            frame_times = [center - 0.5, center + 0.5]
            frames = []
            for ft in frame_times:
                idx = int(round(np.clip(ft, 0.0, max(0.0, duration - 0.01)) * fps))
                if frame_count > 0:
                    idx = min(idx, frame_count - 1)
                frame = read_frame(cap, idx, fallback)
                fallback = frame
                frames.append(frame)
            arr = np.stack(frames, axis=0).astype("float32") / 255.0
            ten = torch.from_numpy(arr).permute(3, 0, 1, 2)
            ten = normalize_video(resize_short_side(ten, 224))
            clips.extend(spatial_crops(ten, 224))
    finally:
        cap.release()
    return torch.stack(clips, dim=0).to(device)


def load_imu_window(csv_path: str, timestamp_sec: float, order: str, target_len: int = 2000) -> np.ndarray:
    cols = ["canonical_timestamp_ms", "gyro_x", "gyro_y", "gyro_z", "accl_x", "accl_y", "accl_z"]
    df = pd.read_csv(csv_path, usecols=cols)
    ts = df["canonical_timestamp_ms"].to_numpy(dtype=np.float64) / 1000.0
    values = df[["gyro_x", "gyro_y", "gyro_z", "accl_x", "accl_y", "accl_z"]].to_numpy(dtype=np.float32)
    if order == "accel_first":
        values = values[:, [3, 4, 5, 0, 1, 2]]
    elif order != "gyro_first":
        raise ValueError(f"Unknown IMU order: {order}")
    start = float(timestamp_sec) - 5.0
    end = float(timestamp_sec) + 5.0
    mask = (ts >= start) & (ts <= end)
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(ts - float(timestamp_sec))))
        lo = max(0, nearest - target_len // 2)
        hi = min(len(ts), lo + target_len)
        lo = max(0, hi - target_len)
        seg_t = ts[lo:hi]
        seg_v = values[lo:hi]
    else:
        seg_t = ts[mask]
        seg_v = values[mask]
    if len(seg_t) < 2:
        seg_t = np.array([start, end], dtype=np.float64)
        base = seg_v[0] if len(seg_v) else np.zeros(6, dtype=np.float32)
        seg_v = np.stack([base, base], axis=0)
    grid = np.linspace(max(start, float(seg_t[0])), min(end, float(seg_t[-1])), target_len)
    out = np.stack([np.interp(grid, seg_t, seg_v[:, c]).astype(np.float32) for c in range(6)], axis=0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype("float32")


def decode_audio_window(
    video_path: str,
    center_sec: float,
    window_sec: float = 6.0,
    sample_rate: int = 16000,
) -> np.ndarray:
    start_sec = max(0.0, float(center_sec) - window_sec / 2.0)
    end_sec = start_sec + window_sec
    target_samples = int(round(window_sec * sample_rate))
    container = av.open(str(video_path))
    if len(container.streams.audio) == 0:
        container.close()
        raise RuntimeError("missing audio stream")
    stream = container.streams.audio[0]
    resampler = AudioResampler(format="fltp", layout="mono", rate=sample_rate)
    try:
        container.seek(int(start_sec * av.time_base), any_frame=False, backward=True)
    except Exception:
        pass

    chunks: List[np.ndarray] = []
    stop = False
    for packet in container.demux(stream):
        for frame in packet.decode():
            frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else None
            if frame_time is not None and frame_time > end_sec:
                stop = True
                break
            if frame_time is not None:
                frame_duration = frame.samples / float(frame.sample_rate)
                if frame_time + frame_duration < start_sec:
                    continue
            outputs = resampler.resample(frame)
            if not isinstance(outputs, list):
                outputs = [outputs]
            for out in outputs:
                arr = out.to_ndarray()
                if arr.ndim == 2:
                    arr = arr.mean(axis=0)
                chunks.append(arr.astype("float32", copy=False))
        if stop or sum(c.shape[0] for c in chunks) >= target_samples:
            break
    container.close()

    if not chunks:
        raise RuntimeError("no decoded audio samples in window")
    waveform = np.concatenate(chunks).astype("float32", copy=False)
    if waveform.shape[0] < target_samples:
        waveform = np.pad(waveform, (0, target_samples - waveform.shape[0]))
    return waveform[:target_samples]


def load_audio_tensor(
    video_path: str,
    center_sec: float,
    device: torch.device,
    window_sec: float = 6.0,
    sample_rate: int = 16000,
    clips_per_video: int = 3,
    num_mel_bins: int = 128,
    target_length: int = 204,
    mean: float = -4.268,
    std: float = 9.138,
) -> torch.Tensor:
    from imagebind import data as ib_data
    waveform = decode_audio_window(video_path, center_sec, window_sec=window_sec, sample_rate=sample_rate)
    waveform_t = torch.from_numpy(waveform).float().unsqueeze(0)
    clip_samples = max(1, int(round(window_sec * sample_rate / clips_per_video)))
    clips = []
    for clip_idx in range(clips_per_video):
        start = clip_idx * clip_samples
        end = min(start + clip_samples, waveform_t.shape[1])
        clip = waveform_t[:, start:end]
        if clip.shape[1] < clip_samples:
            clip = F.pad(clip, (0, clip_samples - clip.shape[1]))
        mel = ib_data.waveform2melspec(clip, sample_rate, num_mel_bins, target_length)
        clips.append(((mel - mean) / std).to(device))
    return torch.stack(clips, dim=0)


def extract_wav_with_pyav(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> Dict[str, Any]:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    if wav_path.exists() and wav_path.stat().st_size > 44:
        return {"ok": True, "cached": True, "samples": None}
    tmp = wav_path.with_suffix(".tmp.wav")
    if tmp.exists():
        tmp.unlink()
    try:
        container = av.open(str(video_path))
        streams = [s for s in container.streams if s.type == "audio"]
        if not streams:
            return {"ok": False, "reason": "no_audio_stream"}
        stream = streams[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=sample_rate)
        total_samples = 0
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            for packet in container.demux(stream):
                for frame in packet.decode():
                    frames = resampler.resample(frame)
                    if frames is None:
                        continue
                    if not isinstance(frames, list):
                        frames = [frames]
                    for out_frame in frames:
                        arr = out_frame.to_ndarray()
                        arr = np.asarray(arr)
                        if arr.ndim == 2:
                            arr = arr.T.reshape(-1)
                        arr = arr.astype(np.int16, copy=False)
                        total_samples += int(arr.shape[0])
                        wf.writeframes(arr.tobytes())
            frames = resampler.resample(None)
            if frames is not None:
                if not isinstance(frames, list):
                    frames = [frames]
                for out_frame in frames:
                    arr = np.asarray(out_frame.to_ndarray())
                    if arr.ndim == 2:
                        arr = arr.T.reshape(-1)
                    arr = arr.astype(np.int16, copy=False)
                    total_samples += int(arr.shape[0])
                    wf.writeframes(arr.tobytes())
        if total_samples <= 0:
            if tmp.exists():
                tmp.unlink()
            return {"ok": False, "reason": "decoded_zero_samples"}
        tmp.replace(wav_path)
        return {"ok": True, "cached": False, "samples": total_samples}
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        return {"ok": False, "reason": f"{type(exc).__name__}: {str(exc)[:160]}"}


def robust_u16_to_u8(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    vals = arr[finite].astype(np.float32)
    lo, hi = np.percentile(vals, [1, 99]).astype(np.float32)
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    out = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def image_msg_to_pil(msg: Any, force_rgb: bool) -> Image.Image:
    enc = str(getattr(msg, "encoding", "")).lower()
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)
    data = bytes(msg.data)
    big = bool(getattr(msg, "is_bigendian", False))

    if enc in {"rgb8", "bgr8"}:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, step // 3, 3)[:, :w, :]
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
        img = Image.fromarray(arr, mode="RGB")
    elif enc in {"rgba8", "bgra8"}:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, step // 4, 4)[:, :w, :]
        if enc == "bgra8":
            arr = arr[:, :, [2, 1, 0, 3]]
        img = Image.fromarray(arr[:, :, :3], mode="RGB")
    elif enc in {"mono8", "8uc1"}:
        arr = np.frombuffer(data, dtype=np.uint8).reshape(h, step)[:, :w]
        img = Image.fromarray(arr, mode="L")
    elif enc in {"mono16", "16uc1"}:
        dtype = ">u2" if big else "<u2"
        arr = np.frombuffer(data, dtype=dtype).reshape(h, step // 2)[:, :w]
        img = Image.fromarray(robust_u16_to_u8(arr), mode="L")
    else:
        raise RuntimeError(f"Unsupported image encoding: {enc}")

    return img.convert("RGB") if force_rgb else img


def imu_window_from_arrays(
    ts: np.ndarray,
    values: np.ndarray,
    center_sec: float,
    order: str,
    target_len: int,
    half_window_sec: float,
) -> np.ndarray:
    if order == "accel_first":
        values = values[:, [3, 4, 5, 0, 1, 2]]
    elif order != "gyro_first":
        raise ValueError(f"Unknown IMU order: {order}")

    start = center_sec - half_window_sec
    end = center_sec + half_window_sec
    mask = (ts >= start) & (ts <= end)
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(ts - center_sec)))
        width = min(len(ts), target_len)
        lo = max(0, nearest - width // 2)
        hi = min(len(ts), lo + width)
        lo = max(0, hi - width)
        seg_t = ts[lo:hi]
        seg_v = values[lo:hi]
    else:
        seg_t = ts[mask]
        seg_v = values[mask]

    if len(seg_t) < 2:
        seg_t = np.array([start, end], dtype=np.float64)
        base = seg_v[0] if len(seg_v) else np.zeros(6, dtype=np.float32)
        seg_v = np.stack([base, base], axis=0)

    grid = np.linspace(max(start, float(seg_t[0])), min(end, float(seg_t[-1])), target_len)
    out = np.stack([np.interp(grid, seg_t, seg_v[:, c]).astype(np.float32) for c in range(6)], axis=0)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def write_video(path: Path, frame_paths: Sequence[str], fps: float = 4.0) -> None:
    import cv2

    frames = [cv2.imread(str(value), cv2.IMREAD_COLOR) for value in frame_paths]
    if any(frame is None for frame in frames):
        raise RuntimeError(f"failed to read a video frame for {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer: {path}")
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()


if __name__ == '__main__':
    main()
