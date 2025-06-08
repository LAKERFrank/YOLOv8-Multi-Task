import torch.nn as nn

from ultralytics.tracknet.tracknet_v4 import TrackNetV4Model
from ultralytics.nn.tasks import PoseModel, yaml_model_load


class MultiTaskModel(nn.Module):
    """Wrapper that holds a TrackNet model and a YOLO pose model."""

    def __init__(self, cfg=None, track_cfg='yolov8n.yaml', pose_cfg='yolov8n-pose.yaml', verbose=True):
        """Initialize TrackNet and Pose models from YAML paths or dicts."""
        super().__init__()
        if cfg:
            if not isinstance(cfg, dict):
                cfg = yaml_model_load(cfg)
            track_cfg = cfg.get('track', track_cfg)
            pose_cfg = cfg.get('pose', pose_cfg)
        self.track = TrackNetV4Model(track_cfg, verbose=verbose)
        self.pose = PoseModel(pose_cfg, verbose=verbose)

    def forward(self, track_input, pose_input, *args, **kwargs):
        track_out = self.track(track_input, *args, **kwargs)
        pose_out = self.pose(pose_input, *args, **kwargs)
        return track_out, pose_out


