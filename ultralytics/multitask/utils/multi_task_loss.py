from __future__ import annotations

import torch

from ultralytics.tracknet.utils.confusion_matrix import ConfConfusionMatrix
from .loss import TrackNetLoss
from ultralytics.yolo.utils.loss import v8PoseLoss


class MultiTaskLoss:
    """Combine TrackNetLoss and YOLO Pose loss."""

    def __init__(self, model) -> None:
        self.track_loss = TrackNetLoss(model.model[model.detect_idx], model.args)
        self.pose_loss = v8PoseLoss(model.model[model.pose_idx], model.args)

    def init_conf_confusion(self, cm: ConfConfusionMatrix) -> None:
        self.track_loss.init_conf_confusion(cm)

    def __call__(self, preds, batch):
        track_pred, pose_pred = preds
        loss_t, items_t = self.track_loss(track_pred, batch)
        loss_p, items_p = self.pose_loss(pose_pred, batch)
        return loss_t + loss_p, torch.cat((items_t, items_p))

