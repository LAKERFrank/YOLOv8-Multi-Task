import argparse
from ultralytics.multitask.train import MultiTaskTrainer


def main(arg):
    overrides = {
        'model': arg.model_path,
        'data': arg.data,
        'epochs': arg.epochs,
        'plots': arg.plots,
        'batch': arg.batch,
        'patience': 300,
        'mosaic': 0.0,
        'val': arg.val,
        'use_dxdy_loss': arg.use_dxdy_loss,
        'use_resampler': arg.use_resampler,
        'save_period': 10,
    }

    if arg.mode == 'train':
        trainer = MultiTaskTrainer(overrides=overrides)
        trainer.train()
    else:
        raise ValueError(f"Unsupported mode: {arg.mode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the MultiTask model with overrides.")
    parser.add_argument('--model_path', type=str,
                        default='ultralytics/models/v8/multitask.yaml',
                        help='Path to the model config or weights')
    parser.add_argument('--mode', type=str, default='train', help='Running mode')
    parser.add_argument('--data', type=str, default='multitask.yaml',
                        help='Dataset configuration')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--plots', action='store_true', help='Save plots during training')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--val', action='store_true', default=True, help='Run validation')
    parser.add_argument('--use_dxdy_loss', action='store_true', default=True, help='Use dxdy loss')
    parser.add_argument('--use_resampler', action='store_true', default=True, help='Use resampler on each epoch')

    args = parser.parse_args()
    main(args)
