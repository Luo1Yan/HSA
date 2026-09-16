"""Protocol and download boundary tests; no original datasets or GPU required."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import download
import extract
import protocols


class ProtocolTests(unittest.TestCase):
    def test_protocol_identity_is_independent_of_json_formatting(self):
        recipe = protocols.experiments(dataset="utd-mhad")[0]
        expected = protocols.protocol_digest(recipe)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for name in (recipe["collection"] + ".json", "text_templates.json", "downloads.json"):
                data = protocols.read_json(protocols.CONFIG_ROOT / name)
                (root / name).write_bytes(json.dumps(data, indent=2).replace("\n", "\r\n").encode())
            with patch.object(protocols, "CONFIG_ROOT", root):
                self.assertEqual(protocols.protocol_digest(recipe), expected)

    def test_complete_inventory_and_observation_identity(self):
        rows = protocols.experiments()
        self.assertEqual(len(rows), 30)
        self.assertEqual(sum(r["task"] == "retrieval" for r in rows), 19)
        self.assertEqual(len({protocols.experiment_id(r) for r in rows}), 30)
        for row in rows:
            with self.subTest(experiment=protocols.experiment_id(row)):
                records = protocols.read_json(protocols.CONFIG_ROOT / (row["collection"] + ".json"))["records"]
                ids = {r["id"] for r in records}
                train, test = row["train_ids"], row["test_ids"]
                self.assertEqual(len(train), len(set(train)))
                self.assertEqual(len(test), len(set(test)))
                self.assertFalse(set(train) & set(test))
                self.assertTrue(set(train + test) <= ids)
                if row["task"] == "classification":
                    self.assertEqual(len(row["train_labels"]), len(train))
                    self.assertEqual(len(row["test_labels"]), len(test))
                    self.assertEqual(set(row["train_labels"]), set(range(len(row["class_names"]))))
                if "text_templates" in row:
                    templates = protocols.read_json(protocols.CONFIG_ROOT / "text_templates.json")[row["text_templates"]]
                    self.assertEqual(len(templates), 80)
                    self.assertTrue(all(t.count("{}") == 1 for t in templates))

    def test_prototypes_use_training_rows_only(self):
        recipe = dict(task="classification", a="depth", hub="image", b="imu",
                      train_ids=["a", "b", "c", "d"], test_ids=["x", "y"],
                      train_labels=[0, 0, 1, 1], test_labels=[0, 1], class_names=["one", "two"])
        vectors = dict(a=[1., 0.], b=[0., 1.], c=[-1., 0.], d=[0., -1.], x=[1., 1.], y=[-1., -1.])
        features = {(key, modality): np.array(value) for key, value in vectors.items() for modality in ["depth", "image", "imu"]}
        first = extract.assemble(recipe, features)
        recipe["test_labels"] = [1, 0]
        features[("x", "imu")] = np.array([999., -999.])
        second = extract.assemble(recipe, features)
        np.testing.assert_array_equal(first["prototypes"], second["prototypes"])
        np.testing.assert_allclose(first["prototypes"], [[2**-0.5, 2**-0.5], [-2**-0.5, -2**-0.5]])

    def test_cached_observations_are_shared_across_tasks(self):
        with tempfile.TemporaryDirectory() as root:
            retrieval, classification = protocols.experiments(dataset="utd-mhad")
            first = extract.FeatureStore(root, retrieval, 8)
            first.put(["a1_s1_t1"], "depth", np.array([[1., 2., 3.]], np.float32))
            first.close()
            second = extract.FeatureStore(root, classification, 4)
            np.testing.assert_array_equal(second.get("a1_s1_t1", "depth"), [1, 2, 3])
            second.close()

    def test_mavd_rgb_is_not_confused_with_same_named_depth(self):
        with tempfile.TemporaryDirectory() as root:
            base = Path(root) / "raw/mavd/drive_example"
            for folder in ("fl_rgb", "fl_rgb_depth"):
                path = base / folder / "fl_rgb_123.jpg"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"only the path is needed for preflight")
            row = {"id": "drive_example/123"}
            adapter = extract.RawInputs(root, "mavd", [row])
            self.assertEqual(adapter.check(row, "image"), base / "fl_rgb/fl_rgb_123.jpg")


class ArchiveTests(unittest.TestCase):
    def test_archive_cannot_write_outside_destination(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            path = root / "bad.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../escaped.txt", "bad")
            with self.assertRaises(ValueError):
                download.unpack(path, root / "data")
            self.assertFalse((root / "escaped.txt").exists())

    def test_multipart_stream_reconstructs_original_tar(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
                info = tarfile.TarInfo("videos/example.txt")
                info.size = 5
                archive.addfile(info, io.BytesIO(b"hello"))
            data = buffer.getvalue()
            paths = [root / f"part-{i}" for i in range(3)]
            for i, path in enumerate(paths):
                path.write_bytes(data[len(data)*i//3:len(data)*(i+1)//3])
            download.unpack_parts(paths, root / "raw")
            self.assertEqual((root / "raw/videos/example.txt").read_bytes(), b"hello")


class DownloadTests(unittest.TestCase):
    def test_server_ignoring_range_restarts_and_validates(self):
        payload = b"registered data\n" * 100
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as root:
                target = Path(root) / "sample.bin"
                target.with_suffix(".bin.part").write_bytes(b"incomplete old data")
                url = f"http://127.0.0.1:{server.server_port}/data"
                download.fetch_file(url, target, hashlib.sha256(payload).hexdigest(), len(payload))
                self.assertEqual(target.read_bytes(), payload)
                target.write_bytes(b"corrupted")
                with self.assertRaises(ValueError):
                    download.fetch_file(url, target, hashlib.sha256(payload).hexdigest(), len(payload))
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
