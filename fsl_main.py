"""
Example for running few-shot algorithms with the PyTorch Lightning wrappers.
"""

import learn2learn as l2l
import pytorch_lightning as pl
from argparse import ArgumentParser
from learn2learn.algorithms import (
    LightningPrototypicalNetworks,
    LightningMetaOptNet,
    LightningMAML,
    LightningANIL,
)
from learn2learn.utils.lightning import EpisodicBatcher

from project.models.scnn4 import SCNN4
from project.utils.lightning import TrackTestAccuracyCallback


def main():
    parser = ArgumentParser(conflict_handler="resolve", add_help=True)
    # add model and trainer specific args
    parser = LightningPrototypicalNetworks.add_model_specific_args(parser)
    parser = LightningMetaOptNet.add_model_specific_args(parser)
    parser = LightningMAML.add_model_specific_args(parser)
    parser = LightningANIL.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    # add script-specific args
    parser.add_argument("--algorithm", type=str, default="protonet")
    parser.add_argument('--neural_coding', type=str, choices=["rate", "ttfs", "phase", "saccade", None], default=None)
    parser.add_argument('--is_grayscale', action='store_true')
    parser.add_argument('--timesteps', type=int, default=12)
    parser.add_argument("--dataset", type=str, default="mini-imagenet",
                        choices=["mini-imagenet", "tiered-imagenet", "cifarfs", "omniglot", "fc100"])
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--meta_batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dict_args = vars(args)

    pl.seed_everything(args.seed)

    # Create tasksets using the benchmark interface
    if False and args.dataset in ["mini-imagenet", "tiered-imagenet"]:
        data_augmentation = "lee2019"
    else:
        data_augmentation = "normalize"
    tasksets = l2l.vision.benchmarks.get_tasksets(
        name=args.dataset,
        train_samples=args.train_queries + args.train_shots,
        train_ways=args.train_ways,
        test_samples=args.test_queries + args.test_shots,
        test_ways=args.test_ways,
        root=args.root,
        data_augmentation=data_augmentation,
    )
    episodic_data = EpisodicBatcher(
        tasksets.train,
        tasksets.validation,
        tasksets.test,
        epoch_length=args.meta_batch_size * 10,
    )

    # init model
    if args.dataset in ["mini-imagenet", "tiered-imagenet"]:
        model = l2l.vision.models.ResNet12(output_size=args.train_ways)
    else:  # CIFAR-FS, FC100
        # model = l2l.vision.models.CNN4(
        #     output_size=args.train_ways,
        #     hidden_size=64,
        #     embedding_size=64 * 4,
        # )
        model = SCNN4(
            output_size=args.train_ways,
            timesteps=args.timesteps,
            channels=1 if args.is_grayscale else 3,
            hidden_size=64,
            embedding_size=4 * 64,
            neural_coding=args.neural_coding
        )
    features = model.features
    classifier = model.classifier

    # init algorithm
    if args.algorithm == "protonet":
        algorithm = LightningPrototypicalNetworks(features=features, **dict_args)
    elif args.algorithm == "maml":
        algorithm = LightningMAML(model, **dict_args)
    elif args.algorithm == "anil":
        algorithm = LightningANIL(features, classifier, **dict_args)
    elif args.algorithm == "metaoptnet":
        algorithm = LightningMetaOptNet(features, **dict_args)

    trainer = pl.Trainer.from_argparse_args(
        args,
        gpus=1,
        accumulate_grad_batches=args.meta_batch_size,
        callbacks=[
            TrackTestAccuracyCallback(episodic_data), #l2l.utils.lightning.TrackTestAccuracyCallback(),
            l2l.utils.lightning.NoLeaveProgressBar(),
        ],
    )
    
    trainer.fit(model=algorithm, datamodule=episodic_data)
    trainer.test(ckpt_path="best")


if __name__ == "__main__":
    main()
