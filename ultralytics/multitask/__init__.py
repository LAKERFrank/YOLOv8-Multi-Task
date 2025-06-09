# Ultralytics MultiTask 🚀, AGPL-3.0 license

"""Convenience imports for the MultiTask package."""

from .engine.model import TrackNet as MultiTask
from .multitask import MultiTaskModel
from .train import TrackNetTrainer as MultiTaskTrainer
from .val import TrackNetValidator as MultiTaskValidator
from .predict import TrackNetPredictor as MultiTaskPredictor

__all__ = (
    'MultiTask',
    'MultiTaskModel',
    'MultiTaskTrainer',
    'MultiTaskValidator',
    'MultiTaskPredictor',
)
