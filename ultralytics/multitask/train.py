"""Training utilities for the MultiTask model."""

from copy import copy

from ultralytics.tracknet.configurable_dataset import TrackNetConfigurableDataset
from ultralytics.tracknet.tracknet_v4 import TrackNetV4Model
from ultralytics.tracknet.val import TrackNetValidator
from ultralytics.multitask.val import MultiTaskValidator
from ultralytics.tracknet.val_dataset import TrackNetValDataset
from ultralytics.yolo.utils import DEFAULT_CFG, RANK
from ultralytics.yolo.v8.detect.train import DetectionTrainer

from .multitask import MultiTaskModel


class TrackNetTrainer(DetectionTrainer):
    """Trainer that operates on TrackNet datasets."""

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "train":
            return TrackNetConfigurableDataset(root_dir=img_path)
        return TrackNetValDataset(root_dir=img_path)

    def get_model(self, cfg=None, weights=None, verbose=True):
        self.tracknet_model = TrackNetV4Model(cfg, ch=10, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            self.tracknet_model.load(weights)
        return self.tracknet_model

    def preprocess_batch(self, batch):
        batch["img"] = batch["img"].to(self.device, non_blocking=True).float() / 255
        for k in ["target"]:
            batch[k] = batch[k].to(self.device)
        return batch

    def get_validator(self):
        self.loss_names = "pos_loss", "conf_loss"
        return TrackNetValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def progress_string(self):
        self.add_callback("print_confusion_matrix", self.tracknet_model.print_confusion_matrix())
        self.add_callback("init_conf_confusion", self.tracknet_model.init_conf_confusion())
        return ("\n" + "%11s" * (3 + len(self.loss_names))) % ("Epoch", "GPU_mem", *self.loss_names, "Size")

    def plot_training_samples(self, batch, ni):
        pass

    def plot_training_labels(self):
        pass


class MultiTaskTrainer(TrackNetTrainer):
    """Trainer that trains TrackNet and Pose tasks together."""

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = MultiTaskModel(cfg, verbose=verbose)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = (
            "pos_loss",
            "conf_loss",
            "box_loss",
            "pose_loss",
            "kobj_loss",
            "cls_loss",
            "dfl_loss",
        )
        return MultiTaskValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def progress_string(self):
        self.add_callback("print_confusion_matrix", self.model.print_confusion_matrix())
        self.add_callback("init_conf_confusion", self.model.init_conf_confusion())
        return ("\n" + "%11s" * (3 + len(self.loss_names))) % ("Epoch", "GPU_mem", *self.loss_names, "Size")


def train(cfg=DEFAULT_CFG, use_python=False):
    """Launch MultiTask training from the command line."""
    model = cfg.model or "yolov8n.yaml"
    data = cfg.data or "data.yaml"
    device = cfg.device if cfg.device is not None else ""

    args = {"model": model, "data": data, "device": device}
    if use_python:
        from ultralytics import YOLO

        YOLO(model).train(**args)
    else:
        trainer = MultiTaskTrainer(overrides=args)
        trainer.train()


if __name__ == "__main__":
    train()
