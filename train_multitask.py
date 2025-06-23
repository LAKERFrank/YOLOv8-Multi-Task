#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""YOLO 多工任務訓練腳本。

此腳本等同於執行下列指令::

    yolo train \
        model=./ultralytics/multitask/weights/multi_11ch_merged.pt \
        data=multi.yaml \
        epochs=300 imgsz=640 device=0 \
        lr0=0.01 optimizer=SGD \
        freeze_layers=3

可依需求調整路徑及其他超參數。
"""

import argparse
from ultralytics.tracknet.engine.model import TrackNet


def parse_args() -> argparse.Namespace:
    """解析指令列參數。"""
    parser = argparse.ArgumentParser(description="Train YOLO multi-task model")
    parser.add_argument(
        "--model",
        default="./ultralytics/multitask/weights/multi_11ch_merged.pt",
        help="模型權重或配置路徑",
    )
    parser.add_argument("--data", default="multi.yaml", help="資料設定檔")
    parser.add_argument("--epochs", type=int, default=300, help="訓練週期數")
    parser.add_argument("--imgsz", type=int, default=640, help="輸入尺寸")
    parser.add_argument("--device", default=0, help="使用的裝置")
    parser.add_argument("--lr0", type=float, default=0.01, help="初始學習率")
    parser.add_argument("--optimizer", default="SGD", help="最佳化器")
    parser.add_argument(
        "--freeze-layers",
        type=int,
        default=3,
        help="凍結最前層數量",
    )
    return parser.parse_args()


def main() -> None:
    """開始訓練。"""
    args = parse_args()

    overrides = {"model": args.model}
    model = TrackNet(overrides)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=args.device,
        lr0=args.lr0,
        optimizer=args.optimizer,
        freeze_layers=args.freeze_layers,
    )


if __name__ == "__main__":
    main()
