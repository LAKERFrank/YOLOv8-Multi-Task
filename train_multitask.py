#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_multitask.py
==================
使用 Ultralytics YOLO 進行多工任務訓練的簡易腳本。相當於執行：

    yolo train \
        model=./ultralytics/multitask/weights/multi_11ch_merged.pt \
        data=multi.yaml \
        epochs=300 imgsz=640 device=0 \
        lr0=0.01 optimizer=SGD \
        freeze=0-2

依需求可自行調整路徑及其他超參數。
"""

from ultralytics import YOLO


def main() -> None:
    """開始訓練。"""
    model_path = "./ultralytics/multitask/weights/multi_11ch_merged.pt"
    data_path = "multi.yaml"  # 你的多工任務資料設定

    model = YOLO(model_path)
    model.train(
        data=data_path,
        epochs=300,
        imgsz=640,
        device=0,
        lr0=0.01,
        optimizer="SGD",
        freeze="0-2",
    )


if __name__ == "__main__":
    main()
