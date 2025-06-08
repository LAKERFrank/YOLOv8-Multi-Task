# Ultralytics MultiTask YOLO integration

from pathlib import Path
from typing import Any, Dict, Optional

from ultralytics import YOLO
from ultralytics.tracknet.engine.model import TrackNet
from ultralytics.tracknet.train import TrackNetTrainer
from ultralytics.tracknet.val import TrackNetValidator
from ultralytics.tracknet.predict import TrackNetPredictor
from ultralytics.yolo.v8.pose import PoseTrainer, PoseValidator, PosePredictor


class MultiTaskModel:
    """Wrapper holding TrackNet and YOLO pose models."""

    def __init__(
        self,
        tracknet_overrides: Dict[str, Any],
        pose_model: str = "ultralytics/models/v8/multitask.yaml",
        **kwargs: Any,
    ) -> None:
        self.tracknet = TrackNet(tracknet_overrides, **kwargs)
        self.pose = YOLO(pose_model, task="pose")

    def predict(self, source=None, stream=False, **kwargs):
        track_res = self.tracknet.predict(source, stream, **kwargs)
        pose_res = self.pose.predict(source, stream, **kwargs)
        return {"tracknet": track_res, "pose": pose_res}

    def train(self, track_overrides=None, pose_overrides=None):
        if track_overrides is None:
            track_overrides = {}
        if pose_overrides is None:
            pose_overrides = {}
        self.tracknet.train(**track_overrides)
        self.pose.train(**pose_overrides)

    def val(self, track_overrides=None, pose_overrides=None):
        if track_overrides is None:
            track_overrides = {}
        if pose_overrides is None:
            pose_overrides = {}
        track_metrics = self.tracknet.val(**track_overrides)
        pose_metrics = self.pose.val(**pose_overrides)
        return {"tracknet": track_metrics, "pose": pose_metrics}


class MultiTaskTrainer:
    """Simple trainer running TrackNetTrainer and PoseTrainer."""

    def __init__(self, track_args: Optional[Dict[str, Any]] = None, pose_args: Optional[Dict[str, Any]] = None):
        self.track_trainer = TrackNetTrainer(overrides=track_args or {})
        self.pose_trainer = PoseTrainer(overrides=pose_args or {})

    def train(self):
        self.track_trainer.train()
        self.pose_trainer.train()


class MultiTaskValidator:
    """Validate both TrackNet and Pose models."""

    def __init__(self, track_args: Optional[Dict[str, Any]] = None, pose_args: Optional[Dict[str, Any]] = None):
        self.track_validator = TrackNetValidator(args=track_args or {})
        self.pose_validator = PoseValidator(args=pose_args or {})

    def __call__(self, multitask_model: "MultiTaskModel"):
        self.track_validator(model=multitask_model.tracknet.model)
        self.pose_validator(model=multitask_model.pose.model)


class MultiTaskPredictor:
    """Run prediction with both TrackNet and Pose models."""

    def __init__(self, track_model: MultiTaskModel):
        self.tracknet = track_model.tracknet
        self.pose = track_model.pose

    def __call__(self, source=None, stream=False, **kwargs):
        track_res = self.tracknet.predict(source, stream, **kwargs)
        pose_res = self.pose.predict(source, stream, **kwargs)
        return {"tracknet": track_res, "pose": pose_res}


TASK_MAP = {
    "multitask": [MultiTaskModel, MultiTaskTrainer, MultiTaskValidator, MultiTaskPredictor]
}


class MultiTask(MultiTaskModel):
    """Entry point mirroring TrackNet API."""

    def __init__(self, overrides: Dict[str, Any], pose_model: str = "ultralytics/models/v8/multitask.yaml", **kwargs: Any) -> None:
        super().__init__(overrides, pose_model, **kwargs)

