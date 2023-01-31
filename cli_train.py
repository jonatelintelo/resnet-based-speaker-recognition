#!/usr/bin/env python3
########################################################################################
#
# Implement the command-line interface for training a network.
#
# Author(s): Nik Vaessen
########################################################################################

import os
import pathlib

from datetime import datetime

import click
import pytorch_lightning
from lightning_fabric.plugins.environments import SLURMEnvironment

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from skeleton.data.tiny_voxceleb import TinyVoxcelebDataModule
from skeleton.models.prototype import PrototypeSpeakerRecognitionModule

########################################################################################
# entrypoint of script


@click.command()
@click.option(
    "--shard_folder",
    type=pathlib.Path,
    required=True,
    help="path to root folder containing train, val and dev shard subfolders",
)
@click.option(
    "--val_trials_path",
    type=pathlib.Path,
    required=True,
    help="path to .txt file containing val trials",
)
@click.option(
    "--dev_trials_path",
    type=pathlib.Path,
    required=True,
    help="path to .txt file containing dev trials",
)
@click.option(
    "--batch_size",
    type=int,
    default=128,
    help="batch size to use for train and val split",
)
@click.option(
    "--audio_length_num_frames",
    type=int,
    default=48_000,
    help="The length of each audio file during training in number of frames."
    " All audio is 16Khz, so 1 second is 16k frames.",
)
@click.option(
    "--n_mfcc", type=int, default=40, help="Number of mfc coefficients to retain"
)
@click.option(
    "--embedding_size",
    type=int,
    default=128,
    help="dimensionality of learned speaker embeddings",
)
@click.option(
    "--learning_rate",
    type=float,
    default=3e-3,
    help="constant learning rate used during training",
)
@click.option("--epochs", type=int, default=30, help="number of epochs to train for")
@click.option("--use_gpu", type=bool, default=True, help="whether to use a gpu")
@click.option("--random_seed", type=int, default=1337, help="the random seed")
@click.option(
    "--num_workers", type=int, default=1, help="number of works to use for data loading"
)
def main(
    shard_folder: pathlib.Path,
    val_trials_path: pathlib.Path,
    dev_trials_path: pathlib.Path,
    batch_size: int,
    audio_length_num_frames: int,
    n_mfcc: int,
    embedding_size: int,
    learning_rate: float,
    epochs: int,
    use_gpu: int,
    random_seed: int,
    num_workers: int,
):
    # log input
    print("### input arguments ###")
    print(f"shard_folder={shard_folder}")
    print(f"val_trials_path={val_trials_path}")
    print(f"dev_trials_path={dev_trials_path}")
    print(f"batch_size={batch_size}")
    print(f"audio_length_num_frames={audio_length_num_frames}")
    print(f"n_mfcc={n_mfcc}")
    print(f"embedding_size={embedding_size}")
    print(f"learning_rate={learning_rate}")
    print(f"epochs={epochs}")
    print(f"use_gpu={use_gpu}")
    print(f"random_seed={random_seed}")
    print()

    # set random seed
    pytorch_lightning.seed_everything(random_seed)

    # build data loader
    dm = TinyVoxcelebDataModule(
        shard_folder=shard_folder,
        val_trials_path=val_trials_path,
        dev_trials_path=dev_trials_path,
        batch_size=batch_size,
        chunk_length_num_frames=audio_length_num_frames,
        num_workers=num_workers,
        n_mfcc=n_mfcc,
    )

    # build model
    model = PrototypeSpeakerRecognitionModule(
        num_inp_features=n_mfcc,
        num_embedding=embedding_size,
        num_speakers=100,
        learning_rate=learning_rate,
        val_trials=dm.val_trials,
        test_trials=dm.dev_trials,
    )

    # configure callback managing checkpoints, and checkpoint file names
    pattern = "epoch_{epoch:04d}.step_{step:09d}.val-eer_{val_eer:.4f}"
    ModelCheckpoint.CHECKPOINT_NAME_LAST = pattern + ".last"
    checkpointer = ModelCheckpoint(
        monitor="val_eer",
        filename=pattern + ".best",
        save_last=True,
        auto_insert_metric_name=False,
    )

    # initialize trainer
    version = datetime.now().strftime("version_%Y_%m_%d___%H_%M_%S")
    if "SLURM_JOB_ID" in os.environ:
        version += f"___job_id_{os.environ['SLURM_JOB_ID']}"

    tensorboard_logger = TensorBoardLogger(save_dir="logs/", version=version)
    csv_logger = CSVLogger(save_dir="logs/", version=version)

    trainer = pytorch_lightning.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        callbacks=[checkpointer, LearningRateMonitor()],
        logger=[tensorboard_logger, csv_logger],
        default_root_dir="logs",
        plugins=[SLURMEnvironment(auto_requeue=False)],  #
    )

    # train loop
    trainer.fit(model, datamodule=dm)

    # test loop (on dev set)
    model = model.load_from_checkpoint(checkpointer.best_model_path)
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
