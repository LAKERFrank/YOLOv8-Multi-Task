---
comments: true
description: Learn how to train and run inference with the YOLOv8 MultiTask model that outputs detections, tracking IDs and pose keypoints.
keywords: YOLOv8, MultiTask, track, pose, training, inference
---

# MultiTask

The **MultiTask** model combines object detection, tracking and pose estimation in a single network. To enable these features, set the configuration keys `track` and `pose` to `True` when training or running inference.

## Train

Use the `yolo` command to train the model with both tracking and pose heads enabled:

```bash
yolo train model=ultralytics/models/v8/multitask.yaml data=your_data.yaml \
    epochs=100 imgsz=640 track=True pose=True
```

The same can be achieved from Python:

```python
from ultralytics import YOLO

model = YOLO('ultralytics/models/v8/multitask.yaml')
model.train(data='your_data.yaml', epochs=100, imgsz=640, track=True, pose=True)
```

## Inference

After training, run prediction on images:

```bash
yolo predict model=path/to/best.pt source=path/to/image.jpg track=True pose=True
```

Or perform tracking on video streams:

```bash
yolo track model=path/to/best.pt source=path/to/video.mp4 track=True pose=True
```
