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
    python3 merge_weights.py \
        --ckpt_a ./ultralytics/multitask/weights/yolov8n-pose.pt \
        --ckpt_b ./ultralytics/multitask/weights/tracknet1000.pt \
        --yaml   ultralytics/models/v8/yolov8_multi_11ch.yaml \
        --save   ./ultralytics/multitask/weights/multi_11ch_merged.pt
"""
from pathlib import Path
import argparse
import re
import torch
import yaml
from ultralytics.nn.tasks import PoseModel


# ====== 依你的模型結構調整這些鍵名 ======
FIRST_KEY = "model.0.conv.weight"   # 第一層 conv
# detect head conv/bias (yolov8n: cv2/3/4)；若用 s/m/l/x 請確認層號
DETECT_W_KEYS = ["model.24.cv2.weight", "model.24.cv3.weight", "model.24.cv4.weight"]
DETECT_B_KEYS = ["model.24.cv2.bias", "model.24.cv3.bias", "model.24.cv4.bias"]


def auto_detect_keys(*sds):
    """Return detect conv/bias key names present in *all* state dicts."""
    pattern = re.compile(r"model\.(\d+)\.cv3\.\d+\.2\.weight$")
    sets = []
    for sd in sds:
        idx_map = {}
        for k in sd:
            m = pattern.match(k)
            if m:
                idx_map.setdefault(int(m.group(1)), set()).add(k)
        if not idx_map:
            return DETECT_W_KEYS, DETECT_B_KEYS
        idx = max(idx_map)
        sets.append(set(idx_map[idx]))
    common = set.intersection(*sets)
    if not common:
        common = sets[0]
    ws = sorted(common)
    bs = [k.replace("weight", "bias") for k in ws]
    return ws, bs
# =====================================

def load_sd(path: str) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    return ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else ckpt

def _pad_trim(t: torch.Tensor, size: int, dim: int = 0) -> torch.Tensor:
    """Pad with zeros or trim tensor along dim to match size."""
    if t.shape[dim] == size:
        return t.clone()
    if t.shape[dim] > size:
        return t.narrow(dim, 0, size).clone()
    shape = list(t.shape)
    shape[dim] = size
    out = torch.zeros(*shape, device=t.device, dtype=t.dtype)
    slices = [slice(None)] * t.ndim
    slices[dim] = slice(0, t.shape[dim])
    out[tuple(slices)] = t
    return out


def concat_first(w_a, w_b, target_ic):
    """Concatenate first conv weights and match expected input channels."""
    merged = torch.cat([w_a, w_b], dim=1)
    return _pad_trim(merged, target_ic, dim=1)

def merge_backbone(sd_a, sd_b, first_key, target_shape):
    merged = {k: v.clone() for k, v in sd_b.items()}  # 先複製 B
    merged[first_key] = concat_first(sd_a[first_key], sd_b[first_key], target_shape[1])
    # 其餘 shape 相同取平均
    for k in sd_a:
        if k == first_key or k in DETECT_W_KEYS + DETECT_B_KEYS:
            continue
        if k in sd_b and sd_a[k].shape == sd_b[k].shape:
            merged[k] = 0.5 * (sd_a[k] + sd_b[k])
    return merged

# ---------- Detect head 2 類分類 slice 重排 ----------
def rebuild_cls_weight(w_a, w_b, target_shape):
    """Stack track and pose classification weights to match new shape."""
    oc, ic = target_shape[:2]
    wb = _pad_trim(w_b, ic, dim=1)
    wa = _pad_trim(w_a, ic, dim=1)
    merged = torch.cat([wb, wa], dim=0)
    return _pad_trim(merged, oc, dim=0)

def rebuild_cls_bias(b_a, b_b, target_shape):
    oc = target_shape[0]
    merged = torch.cat([b_b, b_a])
    return _pad_trim(merged, oc, dim=0)

def merge_heads(sd_a, sd_b, merged, sd_ref):
    for wk, bk in zip(DETECT_W_KEYS, DETECT_B_KEYS):
        merged[wk] = rebuild_cls_weight(sd_a[wk], sd_b[wk], sd_ref[wk].shape)
        merged[bk] = rebuild_cls_bias(sd_a[bk], sd_b[bk], sd_ref[bk].shape)
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
    ap.add_argument(
        "--yaml",
        required=True,
        help="Path to the model YAML (e.g. ultralytics/models/v8/yolov8_multi_11ch.yaml)",
    )
    ap.add_argument("--save",   default="merged.pt")
    ap.add_argument(
        "--scale",
        default="n",
        choices=list("nslmx"),
        help="Model scale when YAML defines 'scales' (n,s,m,l,x)",
    )
    args = ap.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    cfg_dict.setdefault("scale", args.scale)
    if not (isinstance(cfg_dict, dict) and "backbone" in cfg_dict and "head" in cfg_dict):
        raise ValueError(
            f"{args.yaml} does not appear to be a model config file with 'backbone' and 'head'."
        )

    sd_a, sd_b = load_sd(args.ckpt_a), load_sd(args.ckpt_b)

    global DETECT_W_KEYS, DETECT_B_KEYS
    if DETECT_W_KEYS[0] not in sd_a or DETECT_W_KEYS[0] not in sd_b:
        DETECT_W_KEYS, DETECT_B_KEYS = auto_detect_keys(sd_a, sd_b)
        print(f"Auto-detected detect keys: {DETECT_W_KEYS}")

    model = PoseModel(cfg_dict)
    sd_new = model.state_dict()

    merged = merge_backbone(sd_a, sd_b, FIRST_KEY, sd_new[FIRST_KEY].shape)
    merged = merge_heads(sd_a, sd_b, merged, sd_new)

    sd_new.update(merged)
    model.load_state_dict(sd_new, strict=False)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model}, args.save)
    print(f"✅ merged checkpoint saved to: {args.save}")

if __name__ == "__main__":
    main()
