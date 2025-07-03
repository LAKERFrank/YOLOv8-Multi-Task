#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=====================
用途：
    ckpt_pose (2-ch Pose) + ckpt_track (10-ch TrackNet) → ckpt_multi (12-ch MultiTask)
合併規則：
  1. conv0: concat 第一層權重 → 12 通道
  2. backbone/neck: shape 相同則取平均 (A+B)/2
  3. Detect Head:
       - cls weight/bias 按 index concat (不平均)
       - 其餘 detect head 參數（如 conv, bn）若 shape 同樣平均
  4. Pose Head (kpt): 直接複製 A
"""
import argparse
from pathlib import Path
import re
import torch
import yaml
# from ultralytics import YOLO
from ultralytics.engine.model import Model

# ====== 依實際模型 YAML 調整 ======
FIRST_KEY = "model.0.conv.weight"  # 第一層 conv 權重在 state_dict 裡的 key
DETECT_KEY_PATTERN = re.compile(r"model\.(\d+)\.cv[23]\.(?:weight|bias)$")
# =================================

def load_state_dict(path: str):
    ckpt = torch.load(path, map_location="cpu")
    # 如果是 dict 且包含 "model"，再取出 .model.state_dict()
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"].state_dict()
    else:
        return ckpt if isinstance(ckpt, dict) else ckpt.state_dict()

def _pad_trim(t: torch.Tensor, size: int, dim: int = 1):
    """Pad or trim t along dim to length=size."""
    if t.shape[dim] == size:
        return t.clone()
    if t.shape[dim] > size:
        return t.narrow(dim, 0, size).clone()
    # pad
    new_shape = list(t.shape)
    new_shape[dim] = size
    out = torch.zeros(new_shape, dtype=t.dtype, device=t.device)
    idx = [slice(None)] * t.ndim
    idx[dim] = slice(0, t.shape[dim])
    out[tuple(idx)] = t
    return out

def concat_first(w_pose, w_trk, target_ic):
    """把 pose (2ch) + track (10ch) concat → target_ic"""
    merged = torch.cat([w_trk, w_pose], dim=1)  # track 在前, pose 在後
    return _pad_trim(merged, target_ic, dim=1)

def auto_detect_detect_keys(sd):
    """自動找出 detect head 的 weight/bias key 列表"""
    w_keys, b_keys = [], []
    for k in sd:
        m = DETECT_KEY_PATTERN.match(k)
        if m:
            if k.endswith("weight"):
                w_keys.append(k)
            else:
                b_keys.append(k)
    # 保持順序
    w_keys.sort()
    b_keys = [k.replace("weight","bias") for k in w_keys]
    return w_keys, b_keys

def merge_backbone(sd_pose, sd_trk, first_key, target_shape):
    """合併 backbone/neck 等層"""
    merged = {k: v.clone() for k,v in sd_trk.items()}  # 先複製 B
    # conv0 concat → 12ch
    merged[first_key] = concat_first(
        sd_pose[first_key], sd_trk[first_key], target_ic=target_shape[1]
    )
    # 其餘 shape 相同才平均
    for k in sd_pose:
        if k == first_key:
            continue
        if k in sd_trk and sd_pose[k].shape == sd_trk[k].shape:
            merged[k] = 0.5 * (sd_pose[k] + sd_trk[k])
    return merged

def rebuild_cls_weight(w_pose, w_trk, target_shape):
    """Detect head cls weight: track 在前, pose 在後"""
    oc, ic, kh, kw = target_shape
    wb = _pad_trim(w_trk, ic, dim=1)
    wa = _pad_trim(w_pose, ic, dim=1)
    merged = torch.cat([wb, wa], dim=0)  # 拼接 out_channels
    return _pad_trim(merged, oc, dim=0)

def rebuild_cls_bias(b_pose, b_trk, target_shape):
    """Detect head cls bias: track 在前, pose 在後"""
    oc = target_shape[0]
    merged = torch.cat([b_trk, b_pose], dim=0)
    return _pad_trim(merged, oc, dim=0)

def merge_heads(sd_pose, sd_trk, merged, sd_ref):
    """合併 detect head 與 pose head"""
    # auto detect detect head keys
    w_keys, b_keys = auto_detect_detect_keys(sd_trk)
    # 1) cls weight & bias slice
    for wk, bk in zip(w_keys, b_keys):
        merged[wk] = rebuild_cls_weight(
            sd_pose[wk], sd_trk[wk], target_shape=sd_ref[wk].shape
        )
        merged[bk] = rebuild_cls_bias(
            sd_pose[bk], sd_trk[bk], target_shape=sd_ref[bk].shape
        )
    # 2) pose head 全取 pose ckpt
    for k in sd_pose:
        if "kpt" in k or "kp" in k:
            merged[k] = sd_pose[k].clone()
    return merged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_pose", required=True, help="2-ch Pose ckpt")
    parser.add_argument("--ckpt_trk",  required=True, help="10-ch TrackNet ckpt")
    parser.add_argument("--yaml",      required=True, help="12-ch model YAML")
    parser.add_argument("--save",      default="merged_12ch.pt", help="輸出 ckpt 路徑")
    args = parser.parse_args()

    # load yaml & init model (以 YAML 定義的 in_channels=12)
    with open(args.yaml, "r") as f:
        cfg_dict = yaml.safe_load(f)
    # model = YOLO(args.yaml).model
    model = Model(cfg_dict, verbose=False)
    sd_ref = model.state_dict()  # 參考 shape

    # load ckpts
    sd_pose = load_state_dict(args.ckpt_pose)
    sd_trk  = load_state_dict(args.ckpt_trk)

    # backbone/neck merge
    merged = merge_backbone(sd_pose, sd_trk, FIRST_KEY, sd_ref[FIRST_KEY].shape)
    # heads merge
    merged = merge_heads(sd_pose, sd_trk, merged, sd_ref)

    # update new state_dict & save
    sd_ref.update(merged)
    model.load_state_dict(sd_ref, strict=False)
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model}, args.save)
    print(f"✅ 合併完成，儲存到 {args.save}")

if __name__ == "__main__":
    main()

# 用法：python3 convert_yolo8_multitask_12ch.py --ckpt_pose ./ultralytics/multitask/weights/yolov8n-pose-gray-2ch.pt --ckpt_trk ./ultralytics/multitask/weights/tracknet1000.pt --yaml ./ultralytics/models/v8/yolov8-multi_12ch.yaml