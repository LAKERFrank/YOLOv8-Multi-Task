"""DataLoader factory for multi-task TrackNet + Pose training."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from torch.utils.data import DataLoader

from ultralytics.tracknet.dataset import TrackNetDataset

from multi_dataset import MultiDatasetWrapper, collate_fn

__all__ = ["build_multi_loader"]


def build_multi_loader(cfg: Dict) -> Dict[str, DataLoader]:
    """Build training and validation dataloaders from a config dictionary."""
    datasets_cfg = cfg.get("datasets", {})
    track_cfg = datasets_cfg.get("track", {})
    pose_cfg = datasets_cfg.get("pose", {})

    img_size = cfg.get("img_size", 640)
    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 8)

    track_root = track_cfg.get("root")
    pose_root = pose_cfg.get("label_root", track_root)
    frame_indices = tuple(pose_cfg.get("frame_indices", (1, 6)))

    track_train = TrackNetDataset(root_dir=track_root, prefix="train")
    track_val = TrackNetDataset(root_dir=track_root, prefix="val")

    train_ds = MultiDatasetWrapper(track_train, pose_root, frame_indices, img_size)
    val_ds = MultiDatasetWrapper(track_val, pose_root, frame_indices, img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return {"train": train_loader, "val": val_loader}

