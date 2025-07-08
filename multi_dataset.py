"""Wrapper dataset to combine TrackNet and pose data."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import torch
from torch.utils.data import Dataset, default_collate

from ultralytics.tracknet.dataset import TrackNetDataset

__all__ = ["MultiDatasetWrapper", "collate_fn"]


class MultiDatasetWrapper(Dataset):
    """Combine :class:`TrackNetDataset` samples with pose annotations."""

    def __init__(
        self,
        track_ds: TrackNetDataset,
        pose_label_root: str | Path,
        pose_frame_indices: Tuple[int, int] = (1, 6),
        img_size: int = 640,
        transforms: Callable | None = None,
    ) -> None:
        self.track_ds = track_ds
        self.pose_root = Path(pose_label_root)
        self.pose_frame_indices = pose_frame_indices
        self.img_size = img_size
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.track_ds)

    @staticmethod
    def _pad_resize(img: torch.Tensor, size: int) -> torch.Tensor:
        h, w = img.shape
        max_dim = max(h, w)
        pad_diff = max_dim - min(h, w)
        pad1, pad2 = pad_diff // 2, pad_diff - pad_diff // 2
        if h < w:
            pad = (0, 0, pad1, pad2)
        else:
            pad = (pad1, pad2, 0, 0)
        img = cv2.copyMakeBorder(img, *pad, borderType=cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(img).unsqueeze(0).float() / 255.0

    def _load_pose(self, match: str, video: str, frame: str) -> Dict[str, torch.Tensor]:
        img_path = self.pose_root / match / "frame" / video / frame
        label_path = self.pose_root / match / "pose_label" / video / Path(frame).with_suffix(".txt")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        img_tensor = self._pad_resize(img, self.img_size)

        boxes: List[torch.Tensor] = []
        kpts: List[torch.Tensor] = []
        clses: List[int] = []
        if label_path.is_file():
            with open(label_path) as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            for line in lines:
                vals = [float(v) for v in line.split()]
                clses.append(int(vals[0]))
                bbox = torch.tensor(vals[1:5], dtype=torch.float32)
                keypoints = torch.tensor(vals[5:], dtype=torch.float32).view(17, 3)
                bbox[:2] *= torch.tensor([w, h])
                bbox[2:] *= torch.tensor([w, h])
                keypoints[:, 0] *= w
                keypoints[:, 1] *= h
                max_dim = max(w, h)
                pad = (max_dim - min(w, h)) // 2
                if h < w:
                    bbox[1] += pad
                    keypoints[:, 1] += pad
                else:
                    bbox[0] += pad
                    keypoints[:, 0] += pad
                scale = self.img_size / max_dim
                bbox *= scale
                keypoints[:, :2] *= scale
                bbox /= self.img_size
                keypoints[:, :2] /= self.img_size
                boxes.append(bbox)
                kpts.append(keypoints)
        target_boxes = torch.stack(boxes, 0) if boxes else torch.zeros((0, 4))
        target_kpts = torch.stack(kpts, 0) if kpts else torch.zeros((0, 17, 3))
        target_cls = torch.tensor(clses, dtype=torch.int64)
        return {"img": img_tensor, "boxes": target_boxes, "keypoints": target_kpts, "cls": target_cls}

    def __getitem__(self, idx: int) -> Dict:
        track_sample = self.track_ds[idx]
        meta = self.track_ds.samples[idx]
        match = meta["match_name"]
        video = meta["video_name"]

        imgs = []
        boxes_list = []
        kpts_list = []
        cls_list = []
        for i in self.pose_frame_indices:
            frame = meta["img_files"][i]
            pose = self._load_pose(match, video, frame)
            imgs.append(pose["img"])
            boxes_list.append(pose["boxes"])
            kpts_list.append(pose["keypoints"])
            cls_list.append(pose["cls"])

        pose_dict = {
            "imgs": torch.stack(imgs, 0),
            "targets": {
                "frame_idxs": torch.tensor(self.pose_frame_indices, dtype=torch.long),
                "boxes": boxes_list,
                "keypoints": kpts_list,
                "cls": cls_list,
            },
        }
        return {"track": track_sample, "pose": pose_dict}


def collate_fn(batch: List[Dict]) -> Dict:
    track_batch = default_collate([b["track"] for b in batch])
    pose_imgs = torch.stack([b["pose"]["imgs"] for b in batch], 0)
    frame_idxs = default_collate([b["pose"]["targets"]["frame_idxs"] for b in batch])
    boxes = [b["pose"]["targets"]["boxes"] for b in batch]
    keypoints = [b["pose"]["targets"]["keypoints"] for b in batch]
    cls = [b["pose"]["targets"]["cls"] for b in batch]
    pose_batch = {
        "imgs": pose_imgs,
        "targets": {
            "frame_idxs": frame_idxs,
            "boxes": boxes,
            "keypoints": keypoints,
            "cls": cls,
        },
    }
    return {"track": track_batch, "pose": pose_batch}

