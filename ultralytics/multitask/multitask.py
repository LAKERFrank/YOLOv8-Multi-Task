"""Model definition for the single-network multi-task model."""

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.tracknet.utils.confusion_matrix import ConfConfusionMatrix
from .utils.multi_task_loss import MultiTaskLoss


class MultiTaskModel(DetectionModel):
    """YOLO-based model with shared backbone for TrackNet and Pose tasks."""

    def __init__(self, cfg='ultralytics/models/v8/multitask.yaml', ch=10, nc=None, verbose=True):
        # Placeholder indices are assigned before any forward call to avoid
        # AttributeErrors when ``super().__init__`` performs a dry forward pass
        # to build model metadata.
        self.detect_idx, self.pose_idx = None, None
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # Finalize the indices once the model structure has been created.
        self.detect_idx = len(self.model) - 2
        self.pose_idx = len(self.model) - 1

    def _predict_once(self, x, profile=False, visualize=False):
        # Lazily determine detection and pose layer indices. These may not be
        # known during ``__init__`` as ``super().__init__`` can perform an early
        # forward pass before the model structure is finalized.
        if self.detect_idx is None or self.pose_idx is None:
            self.detect_idx = len(self.model) - 2
            self.pose_idx = len(self.model) - 1

        outputs, y = [None, None], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x if m.i in self.save else None)
            if m.i == self.detect_idx:
                outputs[0] = x[0] if isinstance(x, tuple) else x
                x = x[1] if isinstance(x, tuple) else x
        outputs[1] = x[0] if isinstance(x, tuple) else x  # pose output is last
        return outputs

    def init_criterion(self):
        if not hasattr(self, 'multitask_loss'):
            self.multitask_loss = MultiTaskLoss(self)
        return self.multitask_loss

    def init_conf_confusion(self):
        if not hasattr(self, 'multitask_loss'):
            self.multitask_loss = MultiTaskLoss(self)
        self.multitask_loss.init_conf_confusion(ConfConfusionMatrix())

    def print_confusion_matrix(self):
        if hasattr(self, 'multitask_loss'):
            self.multitask_loss.track_loss.confusion_class.print_confusion_matrix()



