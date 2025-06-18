"""Training utilities for the MultiTask model."""

from copy import copy

from ultralytics.tracknet.configurable_dataset import TrackNetConfigurableDataset
from ultralytics.tracknet.tracknet_v4 import TrackNetV4Model
from ultralytics.tracknet.val import TrackNetValidator
from ultralytics.multitask.val import MultiTaskValidator
from ultralytics.tracknet.val_dataset import TrackNetValDataset
from ultralytics.multitask.configurable_dataset import MultiTaskConfigurableDataset
from ultralytics.multitask.val_dataset import MultiTaskValDataset
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.yolo.utils.plotting import Annotator
from ultralytics.yolo.utils.ops import xywh2xyxy
from ultralytics.yolo.utils.torch_utils import torch_distributed_zero_first
from ultralytics.yolo.data import build_dataloader
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
        """Save a few annotated training images to the run directory."""
        try:
            import cv2
        except Exception as e:
            LOGGER.warning(f"visualization skipped: {e}")
            return

        dataset = self.train_loader.dataset
        imgsz = getattr(dataset, "imgsz", 640)
        batch_size = len(batch["img_files"])
        for i in range(min(batch_size, 4)):
            img_path = batch["img_files"][i][-1]
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            annotator = Annotator(img, line_width=2)
            ball = batch["target"][i, -1]
            if ball[1] == 1:
                bx, by = int(ball[2]), int(ball[3])
                cv2.circle(annotator.im, (bx, by), 5, (0, 0, 255), -1)

            if "batch_idx" in batch:
                # Flatten mask to avoid shape mismatch when indexing tensors
                idx = (batch["batch_idx"] == i).view(-1)
                boxes = batch["bboxes"][idx] * imgsz
                kpts = batch["keypoints"][idx] * imgsz
                for box, kpt in zip(boxes, kpts):
                    xyxy = xywh2xyxy(box.unsqueeze(0))[0].tolist()
                    annotator.box_label(xyxy)
                    annotator.kpts(kpt.view(-1, 3), shape=(imgsz, imgsz))

            fname = self.save_dir / f"train_batch{ni}_{i}.jpg"
            cv2.imwrite(str(fname), annotator.result())

    def plot_training_labels(self):
        pass


class MultiTaskTrainer(TrackNetTrainer):
    """Trainer that trains TrackNet and Pose tasks together."""

    def build_dataset(self, img_path, mode="train", batch=None):
        if mode == "train":
            return MultiTaskConfigurableDataset(root_dir=img_path)
        return MultiTaskValDataset(root_dir=img_path)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train", custom_sampler=None):
        """Construct and return dataloader with empty dataset check."""
        assert mode in ["train", "val"]
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty; check dataset path or contents")
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank, custom_sampler)

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
