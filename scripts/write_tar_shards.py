#!/usr/bin/env python
########################################################################################
#
# This file can be used to generate shards for the tiny-voxceleb dataset.
#
# A shard is a tar file with N directories, each directory containing a single
# wav file (audio data) and a single JSON file (information about wav file).
#
# Author(s): Nik Vaessen
########################################################################################

import io
import json
import pathlib
import tarfile
from typing import List, Tuple, Dict

import click
import numpy
import numpy as np
import pandas
import pandas as pd
import tqdm
import torchaudio

########################################################################################
# method to determine shard setup


def determine_shards(
    data_path: pathlib.Path,
    df_meta: pd.DataFrame,
    files_per_shard: int = 5000,
    seed=123,
):
    # collect all audio files and json metadata as a list of tuples
    print(f"recursively globbing {data_path}...")
    audio_file_pairs = [
        (f, collect_json(f, df_meta)) for f in data_path.rglob("*.wav") if f.is_file()
    ]
    num_files = len(audio_file_pairs)

    if num_files <= 0:
        raise ValueError(f"unable to find any wav files within {data_path}")

    # now we assign a file to a shard by just shuffling the list
    # we sort first for reproducibility with the random seed
    audio_file_pairs = sorted(audio_file_pairs, key=lambda tup: tup[1]["sample_id"])

    rng = numpy.random.default_rng(seed)
    rng.shuffle(audio_file_pairs)

    # save the shards as a list of lists
    num_shards = (num_files // files_per_shard) + 1
    shard_collection = [[] for _ in range(num_shards)]

    # loop until all files are in a shard
    for shard in shard_collection:
        while len(shard) < files_per_shard and len(audio_file_pairs) > 0:
            shard.append(audio_file_pairs.pop())

    return shard_collection


def collect_json(file: pathlib.Path, df_meta: pd.DataFrame):
    # determine the content of the JSON file accompanying the wav file

    # path should be ${voxceleb_folder_path}/wav/speaker_id/youtube_id/utterance_id.wav
    speaker_id = file.parent.parent.name
    youtube_id = file.parent.name
    utterance_id = file.stem

    # information about the file
    tensor, sample_rate = torchaudio.load(str(file))
    assert tensor.shape[0] == 1  # mono audio

    # query classification index
    if speaker_id == "eval":
        class_idx = None
    else:
        speaker_row = df_meta.loc[df_meta["id"] == speaker_id]
        speaker_row = speaker_row.reset_index()
        assert len(speaker_row) == 1

        class_idx = speaker_row.at[0, "idx"]
        if np.isnan(class_idx):  # dev set doesn't have a class index
            class_idx = None
        else:
            class_idx = int(class_idx)

    return {
        "sample_id": f"{speaker_id}/{youtube_id}/{utterance_id}",
        "speaker_id": speaker_id,
        "youtube_id": youtube_id,
        "utterance_id": utterance_id,
        "num_frames": len(tensor.squeeze()),
        "sampling_rate": sample_rate,
        "class_idx": class_idx,
    }


########################################################################################
# method to write shards


def dict_to_json_bytes(d: Dict):
    return json.dumps(d).encode("utf-8")


def filter_tarinfo(ti: tarfile.TarInfo):
    ti.uname = "student"
    ti.gname = "mlip"
    ti.mode = 0o0444  # everyone can read
    ti.mtime = 1672527600  # 2023-01-01 00:00:00

    return ti


def write_shard(tar_file_path: pathlib.Path, shard: List[Tuple[pathlib.Path, Dict]]):
    print(f"writing {tar_file_path}")
    num_digits = len(str(len(shard)))

    with tarfile.TarFile(str(tar_file_path), mode="w") as archive:
        for idx, (audio_path, json_obj) in tqdm.tqdm(enumerate(shard)):
            # key for audio file and json file
            spk_id = json_obj["speaker_id"]
            yt_id = json_obj["youtube_id"]
            utt_id = json_obj["utterance_id"]
            key = f"{idx:0>{num_digits}}/{spk_id}/{yt_id}/{utt_id}"

            # add audio file
            archive.add(str(audio_path), arcname=f"{key}.wav", filter=filter_tarinfo)

            # add json file
            json_obj_str = dict_to_json_bytes(json_obj)

            json_tarinfo = tarfile.TarInfo(f"{key}.json")
            json_tarinfo.size = len(json_obj_str)

            archive.addfile(
                tarinfo=filter_tarinfo(json_tarinfo), fileobj=io.BytesIO(json_obj_str)
            )


########################################################################################
# entrypoint of script


@click.command()
@click.option(
    "--in",
    "in_path",
    required=True,
    type=pathlib.Path,
    help="path to a subset of the tiny-voxceleb dataset",
)
@click.option(
    "--meta",
    "meta_file",
    required=True,
    type=pathlib.Path,
    help="path to meta csv file of tiny-voxceleb",
)
@click.option(
    "--out",
    "out_path",
    required=True,
    type=pathlib.Path,
    help="path to root directory where shards will be written",
)
@click.option(
    "--size",
    default=5000,
    type=int,
    help="number of samples per shard",
)
@click.option(
    "--seed",
    default=123,
    type=int,
    help="seed used in determining data subsets",
)
def main(
    in_path: pathlib.Path,
    out_path: pathlib.Path,
    meta_file: pathlib.Path,
    size: int,
    seed: int,
):
    print(f"sharding directory {in_path } to directory {out_path}")

    # make sure out_path exists
    out_path.mkdir(parents=True, exist_ok=True)

    df_meta = pandas.read_csv(str(meta_file))

    # determine how many shards there will be, and what each shard's content is
    shard_collection = determine_shards(
        in_path, df_meta, files_per_shard=size, seed=seed
    )

    # write shards
    for idx, shard in enumerate(shard_collection):
        suffix = (
            "partial.tar" if len(shard) < size and len(shard_collection) > 1 else "tar"
        )
        archive_path = out_path / f"shard-{idx:>06d}.{suffix}"
        write_shard(archive_path, shard)


if __name__ == "__main__":
    main()
