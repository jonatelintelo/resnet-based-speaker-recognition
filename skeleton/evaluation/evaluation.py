########################################################################################
#
# Implement the functions to compute an equal-error-rate for the speaker recognition
# task based on a list of trials.
#
# Author(s): Nik Vaessen
########################################################################################

import pathlib

from dataclasses import dataclass
from typing import List, Tuple
from warnings import warn

import torch as t
import torch.nn.functional

from skeleton.evaluation.eer import calculate_eer

########################################################################################
# define data structures required for evaluating


@dataclass
class EvaluationPair:
    same_speaker: bool
    sample1_id: str
    sample2_id: str


@dataclass
class EmbeddingSample:
    sample_id: str
    embedding: t.Tensor

    def __post_init__(self):
        if isinstance(self.embedding, t.Tensor):
            self._verify_embedding(self.embedding)
        else:
            raise ValueError(f"unexpected {type(self.embedding)=}")

    @staticmethod
    def _verify_embedding(embedding: t.Tensor):
        if len(embedding.shape) != 1:
            raise ValueError("expected embedding to be 1-dimensional tensor")

########################################################################################
# helper methods to read evaluation trials


def read_test_pairs_file(pairs_file_path: pathlib.Path) -> Tuple[bool, str, str]:
    with pairs_file_path.open("r") as f:
        for line in f.readlines():
            line = line.strip()

            if line.count(" ") < 2:
                continue

            gt, path1, path2 = line.strip().split(" ")

            yield bool(int(gt)) if int(gt) >= 0 else None, path1, path2


def load_evaluation_pairs(file_path: pathlib.Path):
    pairs = []

    for gt, path1, path2 in read_test_pairs_file(file_path):
        utt1id = path1.split(".wav")[0]
        utt2id = path2.split(".wav")[0]

        if path1.count("/") == 2:
            spk1id = path1.split("/")[0]
            spk2id = path2.split("/")[0]

            if gt is not None and (spk1id == spk2id) != gt:
                raise ValueError(f"read gt={gt} for line `{path1} {path2}`")

        pairs.append(EvaluationPair(gt, utt1id, utt2id))

    return pairs

########################################################################################
# implementation of evalauting a trial list with cosine-distance


def evaluate_speaker_trials(
    trials: List[EvaluationPair],
    embeddings: List[EmbeddingSample],
    skip_eer: bool = False,
):
    # create a hashmap for quicker access to samples based on key
    sample_map = {}

    for sample in embeddings:
        if sample.sample_id in sample_map:
            raise ValueError(f"duplicate key {sample.sample_id}")

        sample_map[sample.sample_id] = sample

    # compute a list of ground truth scores and prediction scores
    ground_truth_scores = []
    left_sample = []
    right_sample = []

    for pair in trials:
        if pair.sample1_id not in sample_map or pair.sample2_id not in sample_map:
            warn(f"{pair.sample1_id} or {pair.sample2_id} not in sample_map")
            return {
                "eer": -1,
            }

        s1 = sample_map[pair.sample1_id]
        s2 = sample_map[pair.sample2_id]

        gt = 1 if pair.same_speaker else 0

        ground_truth_scores.append(gt)
        left_sample.append(s1.embedding)
        right_sample.append(s2.embedding)

    # transform list of tensors into a single matrix
    left_sample = t.stack(left_sample)
    right_sample = t.stack(right_sample)

    # compute cosine score between left and right samples (-1 to 1)
    prediction_scores = torch.nn.functional.cosine_similarity(left_sample, right_sample)

    # normalize scores to be between 0 and 1
    prediction_scores = torch.clip((prediction_scores + 1) / 2, 0, 1)
    prediction_scores = prediction_scores.tolist()

    # dictionary to return either 1 or 2 values
    return_dict = {"scores": prediction_scores}

    if skip_eer:
        # for eval set, we can only compute the prediction scores as ground truth
        pass
    else:
        # compute the EER with know ground truth values of each trial
        try:
            eer = calculate_eer(ground_truth_scores, prediction_scores)
        except (ValueError, ZeroDivisionError) as e:
            # if NaN values, we just return a very bad score
            # so that programs relying on result don't crash
            print(f"EER calculation had {e}")
            eer = 1

        return_dict["eer"] = eer

    return return_dict
