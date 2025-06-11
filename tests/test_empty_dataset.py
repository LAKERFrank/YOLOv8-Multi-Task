import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "multitask_train", ROOT / "ultralytics" / "multitask" / "train.py"
)
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
MultiTaskTrainer = train.MultiTaskTrainer


def test_empty_dataset(tmp_path):
    data = {"train": str(tmp_path), "val": str(tmp_path), "nc": 1, "names": ["a"]}
    trainer = MultiTaskTrainer(overrides={"model": "ultralytics/models/v8/yolov8.yaml", "data": data, "epochs": 1})
    with pytest.raises(ValueError, match="Dataset is empty; check dataset path or contents"):
        trainer.get_dataloader(str(tmp_path), batch_size=1, mode="train")
