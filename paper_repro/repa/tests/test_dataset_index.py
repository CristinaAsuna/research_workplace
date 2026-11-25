"""The dataset.json index must give exactly the same file lists and labels as the official os.walk.

Run in the cluster train env (torch, numpy, PIL): python tests/test_dataset_index.py
It also runs under pytest when pytest is installed.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _make_edm2_layout(root, n=2503):
    """Mimic dataset_tools.py output: <idx//1000:05d>/img<idx:08d>.png and img-mean-std-<idx:08d>.npy."""
    image_labels, feature_labels = [], []
    for idx in range(n):
        sub = f"{idx // 1000:05d}"
        image = f"{sub}/img{idx:08d}.png"
        feature = f"{sub}/img-mean-std-{idx:08d}.npy"
        for base, name in (("images", image), ("vae-sd", feature)):
            path = Path(root) / base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
        label = (idx * 7) % 1000
        image_labels.append([image, label])
        feature_labels.append([feature, label])
    # dataset_tools writes labels in index order; shuffle one file to prove order does not matter
    feature_labels.reverse()
    for base, labels in (("images", image_labels), ("vae-sd", feature_labels)):
        (Path(root) / base / "dataset.json").write_text(json.dumps({"labels": labels}))


def test_json_index_matches_walk():
    from repa.dataset import CustomDataset

    with tempfile.TemporaryDirectory() as root:
        _make_edm2_layout(root)
        by_json = CustomDataset(root, index="json")
        by_walk = CustomDataset(root, index="walk")
        assert len(by_json) == len(by_walk) == 2503
        assert by_json.image_fnames == by_walk.image_fnames
        assert by_json.feature_fnames == by_walk.feature_fnames
        assert (by_json.labels == by_walk.labels).all()
        assert by_json.image_fnames[1000] == os.path.normpath("00001/img00001000.png")
        assert by_json.labels[1000] == (1000 * 7) % 1000


def test_unknown_index_rejected():
    from repa.dataset import CustomDataset

    with tempfile.TemporaryDirectory() as root:
        _make_edm2_layout(root, n=3)
        try:
            CustomDataset(root, index="glob")
        except ValueError:
            return
        raise AssertionError("index='glob' should raise ValueError")


if __name__ == "__main__":
    test_json_index_matches_walk()
    test_unknown_index_rejected()
    print("dataset index tests passed")
