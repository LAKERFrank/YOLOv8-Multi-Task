"""Pose dataset for single grayscale image and keypoint label.

This module provides :class:`PoseDataset` used to read a single grayscale frame
and its associated YOLOv8 style pose label. Images are padded to square, resized
and converted to ``torch.Tensor`` in ``[0, 1]`` range. Bounding boxes and
keypoints are returned in normalized ``cxcywh``/``xyv`` format.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import cv2
import torch
from torch.utils.data import Dataset

__all__ = ["PoseDataset"]


class PoseDataset(Dataset):
    """Dataset that loads a single image and its pose annotation."""

    def __init__(self, image_paths: Sequence[str | Path], img_size: int = 640) -> None:
        self.image_paths: List[Path] = [Path(p) for p in image_paths]
        self.img_size = img_size

    def __len__(self) -> int:  # pragma: no cover - thin wrapper
        return len(self.image_paths)

    def _pad_resize(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape
        max_dim = max(h, w)
        pad_diff = max_dim - min(h, w)
        pad1, pad2 = pad_diff // 2, pad_diff - pad_diff // 2
        if h < w:
            pad = (0, 0, pad1, pad2)
        else:
            pad = (pad1, pad2, 0, 0)
        img = cv2.copyMakeBorder(img, *pad, borderType=cv2.BORDER_CONSTANT, value=0)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        return torch.from_numpy(img).unsqueeze(0).float() / 255.0

    def _transform_coords(self, coords: torch.Tensor, img_w: int, img_h: int) -> torch.Tensor:
        max_dim = max(img_w, img_h)
        pad_diff = max_dim - min(img_w, img_h)
        pad = pad_diff // 2
        if img_h < img_w:
            coords[:, 1] += pad
        else:
            coords[:, 0] += pad
        scale = self.img_size / max_dim
        coords[:, :2] *= scale
        coords /= self.img_size
        return coords

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        label_path = img_path.with_suffix(".txt")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        h, w = img.shape
        tensor_img = self._pad_resize(img)

        boxes: List[torch.Tensor] = []
        keypoints: List[torch.Tensor] = []
        classes: List[int] = []

        if label_path.is_file():
            with open(label_path, "r") as f:
                lines = [x.strip() for x in f.readlines() if x.strip()]
            for line in lines:
                vals = [float(v) for v in line.split()]
                cls = int(vals[0])
                bbox = torch.tensor(vals[1:5], dtype=torch.float32)
                kps = torch.tensor(vals[5:], dtype=torch.float32).view(17, 3)
                # convert to pixel coords
                bbox[:2] *= torch.tensor([w, h])
                bbox[2:] *= torch.tensor([w, h])
                kps[:, 0] *= w
                kps[:, 1] *= h
                bbox = self._transform_coords(bbox.view(1, 4), w, h).squeeze(0)
                kps[:, :2] = self._transform_coords(kps[:, :2], w, h)
                boxes.append(bbox)
                keypoints.append(kps)
                classes.append(cls)

        target_boxes = torch.stack(boxes, 0) if boxes else torch.zeros((0, 4))
        target_kpts = torch.stack(keypoints, 0) if keypoints else torch.zeros((0, 17, 3))
        target_cls = torch.tensor(classes, dtype=torch.int64)

        return {
            "img": tensor_img,
            "boxes": target_boxes,
            "keypoints": target_kpts,
            "cls": target_cls,
            "frame_idx": int(img_path.stem),
            "img_path": img_path,
        }

