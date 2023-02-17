#! /usr/bin/env python3
########################################################################################
#
# Implement the command-line interface for prediction/inference.
#
# Author(s): Nik Vaessen, David van Leeuwen
########################################################################################

import argparse
import pathlib
import traceback

from pathlib import Path

import torch as t
import torch.utils.data

from tqdm import tqdm

from skeleton.data.datapipe import (
    construct_sample_datapipe,
    pipe_mfcc,
    pipe_batch_samples,
)
from skeleton.models.prototype import PrototypeSpeakerRecognitionModule

from skeleton.evaluation.evaluation import (
    evaluate_speaker_trials,
    EmbeddingSample,
    load_evaluation_pairs,
)
from skeleton.evaluation.eer import calculate_eer

########################################################################################
# CLI input arguments


parser = argparse.ArgumentParser()

# required positional paths to data
parser.add_argument(
    "checkpoint",
    type=Path,
    help="path to checkpoint file",
)
parser.add_argument(
    "shards_dirs_to_evaluate",
    type=str,
    help="paths to shard file containing all audio files in the given trial list (',' separated)",
)
parser.add_argument(
    "trial_lists", type=str, help="file with list of trials (',' separated)"
)
parser.add_argument("score_file", type=Path, help="output file to store scores")

# optional arguments based on pre-processing variables
parser.add_argument(
    "--n_mfcc",
    type=int,
    default=40,
    help=" Number of mfc coefficients to retain, should be the same as in train",
)
parser.add_argument(
    "--network",
    type=str,
    default="proto",
    help="which network to load from the given checkpoint",
)
parser.add_argument(
    "--use-gpu",
    type=lambda x: x.lower() in ("yes", "true", "t", "1"),
    default=True,
    help="whether to evaluate on a GPU device",
)

################################################################################
# entrypoint of script


def main(
    checkpoint_path: Path,
    shards_dirs_to_evaluate: str,
    trial_lists: str,
    score_file: Path,
    n_mfcc: int,
    network: str,
    use_gpu: bool,
):
    # load module from checkpoint
    if network == "proto":
        try:
            model = PrototypeSpeakerRecognitionModule.load_from_checkpoint(
                str(checkpoint_path), map_location="cpu"
            )
        except RuntimeError:
            traceback.print_exc()
            raise ValueError(
                f"Failed to load PrototypeSpeakerRecognitionModule with "
                f"{checkpoint_path=}. Is {network=} correct?"
            )
    else:
        raise ValueError(f"unknown {network=}")

    # set to eval mode, and move to GPU if applicable
    model = model.eval()

    use_gpu = use_gpu and t.cuda.is_available()
    if use_gpu:
        model = model.to("cuda")

    print(model)
    print(f"{use_gpu=}")

    # split input string if more than 1 test set was given
    shards_dirs = shards_dirs_to_evaluate.split(",")
    trial_lists = trial_lists.split(",")

    # process dev and test simultaneously
    pairs = list()
    embeddings = list()

    for shards_dir, trial_list in zip(shards_dirs, trial_lists):
        print("Processing shards dir", shards_dir)

        # init webdataset pipeline
        # load data pipeline
        dp = construct_sample_datapipe(pathlib.Path(shards_dir), num_workers=1)
        dp = pipe_mfcc(dp, n_mfcc)
        dp = pipe_batch_samples(dp, batch_size=1, drop_last=False)
        loader = torch.utils.data.DataLoader(dp, batch_size=None, num_workers=1)

        # load every trial pair
        pairs.extend(load_evaluation_pairs(Path(trial_list)))

        # loop over all data in the shard and store the speaker embeddings
        with t.no_grad():  # disable autograd as we only do inference
            for sample in tqdm(loader):
                sample_id, network_input, speaker_labels = sample

                assert network_input.shape[0] == 1

                if use_gpu:
                    network_input = network_input.to("cuda")

                # compute embedding
                embedding, _ = model(network_input)

                # determine key (for matching with trial pairs)
                key = sample_id[0]
                if key.startswith("eval"):
                    # for eval, path in shard is eval/wav/$key
                    key = key.split("/")[-1]

                # store embedding, and save on CPU so that GPU memory does not fill up
                embedding = embedding.to("cpu")
                embeddings.append(EmbeddingSample(key, embedding.squeeze()))

    # move model weights back to CPU so that mean_embedding and std_embedding
    # are on same device as embeddings
    model = model.to("cpu")

    # for each trial, compute scores based on cosine similarity between
    # speaker embeddings
    results = evaluate_speaker_trials(pairs, embeddings, skip_eer=True)
    scores = results['scores']

    # write each trial (with computed score) to a file
    eer_scores = list()
    eer_labels = list()
    with open(score_file, "w") as out:
        for score, pair in zip(scores, pairs):
            print(score, f"{pair.sample1_id}", f"{pair.sample2_id}", file=out)
            if pair.same_speaker is not None:
                eer_scores.append(score)
                eer_labels.append(pair.same_speaker)

    # for the dev set, compute EER based on predicted scores and ground truth labels
    if len(eer_scores) > 0:
        eer = calculate_eer(groundtruth_scores=eer_labels, predicted_scores=eer_scores)
        print(
            f"EER computed over {len(eer_scores)} trials for which truth is known: {eer*100:4.2f}%"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(
        checkpoint_path=args.checkpoint,
        shards_dirs_to_evaluate=args.shards_dirs_to_evaluate,
        trial_lists=args.trial_lists,
        score_file=args.score_file,
        n_mfcc=args.n_mfcc,
        network=args.network,
        use_gpu=args.use_gpu,
    )
