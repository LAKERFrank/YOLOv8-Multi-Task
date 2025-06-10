import argparse

from ultralytics.multitask.train import MultiTaskTrainer


def main(arg):
    """Entry point for training the MultiTask model."""

    overrides = {}
    overrides['model'] = arg.model_path
    overrides['mode'] = arg.mode
    overrides['data'] = arg.data
    overrides['epochs'] = arg.epochs
    overrides['plots'] = arg.plots
    overrides['batch'] = arg.batch
    overrides['patience'] = 300
    overrides['mosaic'] = 0.0
    overrides['plots'] = arg.plots
    overrides['val'] = arg.val
    overrides['use_dxdy_loss'] = arg.use_dxdy_loss
    overrides['use_resampler'] = arg.use_resampler
    overrides['save_period'] = 10

    if arg.mode == 'train':
        trainer = MultiTaskTrainer(overrides=overrides)
        trainer.train()
    else:
        raise ValueError(f"Unsupported mode: {arg.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a custom MultiTask model with overrides.")

    parser.add_argument('--model_path', type=str,
                        default='ultralytics/models/v8/multitask.yaml',
                        help='Path to the model')
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode for the training (e.g., train)')
    parser.add_argument('--data', type=str, default='multitask.yaml',
                        help='Data configuration (e.g., multitask.yaml)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of epochs')
    parser.add_argument('--plots', type=bool, default=False,
                        help='Whether to plot or not')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--val', type=bool, default=True, help='Run validation')
    parser.add_argument('--use_dxdy_loss', type=bool, default=True,
                        help='Use dxdy loss or not')
    parser.add_argument('--use_resampler', type=bool, default=True,
                        help='Use resampler on each epoch')

    args = parser.parse_args()
    main(args)
