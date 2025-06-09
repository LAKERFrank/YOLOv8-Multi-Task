"""TrackNet V4 model used by the MultiTask wrapper."""

from ultralytics.tracknet.utils.confusion_matrix import ConfConfusionMatrix
from ultralytics.nn.tasks import DetectionModel

from .utils.loss import TrackNetLoss


class TrackNetV4Model(DetectionModel):
    """DetectionModel wrapper with helpers for TrackNet training."""

    def init_criterion(self):
        """Initialize and return the loss criterion."""
        if not hasattr(self, "track_net_loss"):
            self.track_net_loss = TrackNetLoss(self)
        return self.track_net_loss

    def init_conf_confusion(self):
        """Initialize confusion matrix for confidence calculation."""
        if not hasattr(self, "track_net_loss"):
            self.track_net_loss = TrackNetLoss(self)
        self.track_net_loss.init_conf_confusion(ConfConfusionMatrix())

    def print_confusion_matrix(self):
        """Print the accumulated confusion matrix if available."""
        if not hasattr(self, "track_net_loss"):
            self.track_net_loss = TrackNetLoss(self)
        self.track_net_loss.confusion_class.print_confusion_matrix()
