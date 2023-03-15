########################################################################################
#
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

from typing import List, Optional

import torch.utils.data
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from skeleton.data.datapipe import (
    construct_sample_datapipe,
    pipe_chunk_sample,
    pipe_mfcc,
    pipe_batch_samples,
)
from skeleton.evaluation.evaluation import EvaluationPair, load_evaluation_pairs

########################################################################################
# data module implementation


class TinyVoxcelebDataModule(LightningDataModule):
    def __init__(
        self,
        shard_folder: pathlib.Path,
        batch_size: int,
        chunk_length_num_frames: int,
        val_trials_path: pathlib.Path,
        dev_trials_path: pathlib.Path,
        num_workers: int,
        n_mfcc: int,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.chunk_length_num_frames = chunk_length_num_frames
        self.num_workers_train = num_workers
        self.num_workers_eval = 1 if self.num_workers_train >= 1 else 0
        self.n_mfcc = n_mfcc

        self.shard_folder = shard_folder
        self.val_trials_path = val_trials_path
        self.dev_trials_path = dev_trials_path
        self.is_augmented = True

        # init in setup()
        self.train_dp_original = None
        self.train_dp = None
        self.val_dp = None
        self.dev_dp = None

    def setup(self, stage: Optional[str] = None) -> None:
        # train dataloader (non-augmented)
        train_dp_original = construct_sample_datapipe(not self.is_augmented,
            self.shard_folder / "train", num_workers=self.num_workers_train
        )
        train_dp_original = pipe_chunk_sample(train_dp_original, self.chunk_length_num_frames)
        train_dp_original = pipe_mfcc(train_dp_original, self.n_mfcc)
        train_dp_original = pipe_batch_samples(train_dp_original, self.batch_size, drop_last=True)
        self.train_dp_original = train_dp_original

        # train dataloader (augmented)
        train_dp = construct_sample_datapipe(self.is_augmented,
            self.shard_folder / "train", num_workers=self.num_workers_train
        )
        # train_dp = train_dp_original.concat(train_dp)
        train_dp = pipe_chunk_sample(train_dp, self.chunk_length_num_frames)
        train_dp = pipe_mfcc(train_dp, self.n_mfcc)
        train_dp = pipe_batch_samples(train_dp, self.batch_size, drop_last=True)
        self.train_dp = train_dp

        # self.train_dp = self.train_dp_original.concat(self.train_dp)

        # val dataloader
        val_dp = construct_sample_datapipe(not self.is_augmented,
            self.shard_folder / "val", num_workers=self.num_workers_eval
        )
        val_dp = pipe_chunk_sample(val_dp, self.chunk_length_num_frames)
        val_dp = pipe_mfcc(val_dp, self.n_mfcc)
        val_dp = pipe_batch_samples(val_dp, self.batch_size, drop_last=False)
        self.val_dp = val_dp

        # dev dataloader
        # we explicitly evaluate with a batch size of 1 and the whole utterance
        dev_dp = construct_sample_datapipe(not self.is_augmented,
            self.shard_folder / "dev", num_workers=self.num_workers_eval
        )
        dev_dp = pipe_mfcc(dev_dp, self.n_mfcc)
        dev_dp = pipe_batch_samples(dev_dp, batch_size=1, drop_last=False)
        self.dev_dp = dev_dp

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        self.train_dp = torch.utils.data.ChainDataset([self.train_dp_original, self.train_dp])
        return torch.utils.data.DataLoader(
            self.train_dp, batch_size=None, num_workers=self.num_workers_train
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.val_dp, batch_size=None, num_workers=self.num_workers_eval
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return torch.utils.data.DataLoader(
            self.dev_dp, batch_size=None, num_workers=self.num_workers_eval
        )

    @property
    def val_trials(self) -> List[EvaluationPair]:
        return load_evaluation_pairs(self.val_trials_path)

    @property
    def dev_trials(self) -> List[EvaluationPair]:
        return load_evaluation_pairs(self.dev_trials_path)
