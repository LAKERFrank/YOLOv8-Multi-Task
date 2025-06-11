# Ultralytics MultiTask 🚀, AGPL-3.0 license

"""Convenience imports for the MultiTask package."""

from .engine.model import TrackNet as MultiTask
from .multitask import MultiTaskModel
from .train import MultiTaskTrainer, TrackNetTrainer
from .val import MultiTaskValidator
from .predict import MultiTaskPredictor

__all__ = (
    'MultiTask',
    'MultiTaskModel',
    'MultiTaskTrainer',
    'TrackNetTrainer',
    'MultiTaskValidator',
    'MultiTaskPredictor',
)
