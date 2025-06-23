#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
merge_weights.py
=================
用途：
    ckpt_A (1-通道 Pose) + ckpt_B (10-通道 TrackNet) → ckpt_C (11-通道 Pose+Detect)
策略：
    1. 第一層 conv 權重：dim=1 上 concat，得到 11 通道
    2. Backbone/Neck：shape 完全相同 → (A+B)/2；shape 不同 → 用 B
    3. Detect Head (cls slice)：
         - index 0 = shuttlecock：取 B 的 cls[0]
         - index 1 = person      ：取 A 的 cls[0] (因 A 僅有一類 person)
    4. Pose Head (kpt) 全取 A
使用方式：
    python merge_weights.py \
        --ckpt_a weights/pose_ch1.pt \
        --ckpt_b weights/tracknet10ch.pt \
        --yaml   config/yolov8_multi_11ch.yaml \
        --save   weights/yolov8_multi_11ch_merged.pt
"""
from pathlib import Path
import argparse, torch
from ultralytics import YOLO


# ====== 依你的模型結構調整這些鍵名 ======
FIRST_KEY = "model.0.conv.weight"   # 第一層 conv
# detect head conv/bias (yolov8n: cv2/3/4)；若用 s/m/l/x 請確認層號
DETECT_W_KEYS = ["model.24.cv2.weight", "model.24.cv3.weight", "model.24.cv4.weight"]
DETECT_B_KEYS = ["model.24.cv2.bias",   "model.24.cv3.bias",   "model.24.cv4.bias"]
# =====================================

def load_sd(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt

def concat_first(w_a, w_b):
    """ dim=1 concat: (C_out, 1,3,3) + (C_out, 10,3,3) -> (C_out, 11,3,3) """
    return torch.cat([w_a, w_b], dim=1)

def merge_backbone(sd_a, sd_b, first_key):
    merged = {k: v.clone() for k, v in sd_b.items()}           # 先複製 B
    merged[first_key] = concat_first(sd_a[first_key], sd_b[first_key])
    # 其餘 shape 相同取平均
    for k in sd_a:
        if k == first_key or k in DETECT_W_KEYS + DETECT_B_KEYS:
            continue
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            merged[k] = 0.5 * (sd_a[k] + sd_b[k])
    return merged

# ---------- Detect head 2 類分類 slice 重排 ----------
def rebuild_cls_weight(w_a, w_b):
    # w_a: A 的 detect conv weight，cls slice 只有 1 (person)
    # w_b: B 的 detect conv weight，cls slice 只有 1 (shuttlecock)
    nc_new, in_c = 2, w_b.shape[1]
    new_cls = torch.zeros(nc_new, in_c, 1, 1)
    new_cls[0] = w_b[0].clone()   # shuttlecock
    new_cls[1] = w_a[0].clone()   # person
    tail = w_b[nc_new:]           # bbox/obj/dfl/...
    return torch.cat([new_cls, tail], dim=0)

def rebuild_cls_bias(b_a, b_b):
    new_cls = torch.zeros(2)
    new_cls[0] = b_b[0].clone()   # shuttlecock
    new_cls[1] = b_a[0].clone()   # person
    return torch.cat([new_cls, b_b[2:]], dim=0)  # 同上 tail

def merge_heads(sd_a, sd_b, merged):
    for wk, bk in zip(DETECT_W_KEYS, DETECT_B_KEYS):
        merged[wk] = rebuild_cls_weight(sd_a[wk], sd_b[wk])
        merged[bk] = rebuild_cls_bias(sd_a[bk],  sd_b[bk])
    # Pose 專屬 (包含 'kpt' 關鍵字) → 全取 A
    for k in sd_a:
        if "kpt" in k:
            merged[k] = sd_a[k].clone()
    return merged
# ----------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True)
    ap.add_argument("--ckpt_b", required=True)
    ap.add_argument("--yaml",   required=True)
    ap.add_argument("--save",   default="merged.pt")
    args = ap.parse_args()

    sd_a, sd_b = load_sd(args.ckpt_a), load_sd(args.ckpt_b)
    model = YOLO(args.yaml).model
    sd_new = model.state_dict()

    merged = merge_backbone(sd_a, sd_b, FIRST_KEY)
    merged = merge_heads(sd_a, sd_b, merged)

    sd_new.update(merged)
    model.load_state_dict(sd_new, strict=False)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model}, args.save)
    print(f"✅ merged checkpoint saved to: {args.save}")

if __name__ == "__main__":
    main()
