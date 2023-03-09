########################################################################################
#
# This file implements the components to construct a datapipe for the tiny-voxceleb
# dataset.
#
# Author(s): Nik Vaessen
########################################################################################

import collections
import functools
import json
import pathlib
import random

from typing import Tuple, Dict, List

import torch as t
import torch.utils.data
import torchaudio
import numpy as np

from torch.utils.data.datapipes.utils.common import StreamWrapper
from torchdata.datapipes.iter import (
    FileLister,
    Shuffler,
    Header,
    ShardingFilter,
    FileOpener,
    Mapper,
    TarArchiveLoader,
    WebDataset,
    IterDataPipe,
    Batcher,
)

#import librosa

########################################################################################
# helper methods for decoding binary streams from files to useful python objects

def inject_noise(data, noise_factor):
    noise = torch.randn(len(data))
    augmented_data = data + noise_factor * noise
    
    return augmented_data

def random_speed_change(data, sample_rate):
    speed_factor = random.choice([0.9, 1.0, 1.1])
    if speed_factor == 1.0: # no change
        return data

    # change speed and resample to original rate:
    sox_effects = [
        ["speed", str(speed_factor)],
        ["rate", str(sample_rate)],
    ]
    transformed_audio, _ = torchaudio.sox_effects.apply_effects_tensor(
    data, sample_rate, sox_effects)
    return transformed_audio

def reverb_aug(data,sample_rate): #rir =Room Impulse Respons 
	rir_raw = data
	rir = rir_raw[:, int(sample_rate * 1.01) : 	int(sample_rate * 1.3)]
	rir = rir / torch.norm(rir, p=2)
	RIR = torch.flip(rir, [1])
	pad_data = torch.nn.functional.pad(data, (RIR.shape[1] - 1, 0)) #adds 0 before and after timeseries
	aug= torch.nn.functional.conv1d(pad_data.unsqueeze(0), RIR.unsqueeze(0))[0] # rm none? # (minibatch,in_channels,iW), unsqueeze adds dim so it works with conv
	return aug
 

#def pitch_aug(data, samplingrate, semitones): #only +- 4 or 5 semitones
#    return librosa.effects.pitch_shift(data, samplingrate, semitones)

def random_gain_aug(data, minimum=0.1, maximum=0.12): #change the percieved loudness of the waveform
    gain = random.uniform(minimum, maximum) 
    return data * gain #scale but in amplitude 

def randomize_effect():
    effects = ['inject_noise', 'rd_speed_change','rand_gain', 'reverb', 'none']
    choice = np.random.choice(effects, 1, p=[0.1,0.1,0.1,0.2,0.5]) # if aug on everything then not rep of test dataset
    return choice


def decode_wav(value: StreamWrapper) -> t.Tensor:
    assert isinstance(value, StreamWrapper)
    
    value, sample_rate = torchaudio.load(value)
    choice = randomize_effect()
    if choice == 'inject_noise':
        value = inject_noise(value, 0.01)
    elif choice == 'rd_speed_change':
        value = random_speed_change(value, sample_rate)
    elif choice == 'rand_gain':
        value= random_gain_aug(value, minimum=0.1, maximum=0.12)
    elif choice == 'reverb':
        value= reverb_aug(value,sample_rate)

    assert sample_rate == 16_000

    # make sure that audio has 1 dimension
    value = torch.squeeze(value)

    return value


def decode_json(value: StreamWrapper) -> Dict:
    assert isinstance(value, StreamWrapper)

    return json.load(value)


def decode(element: Tuple[str, StreamWrapper]):
    assert isinstance(element, tuple) and len(element) == 2
    key, value = element

    assert isinstance(key, str)
    assert isinstance(value, StreamWrapper)

    if key.endswith(".wav"):
        value = decode_wav(value)

    if key.endswith(".json"):
        value = decode_json(value)

    return key, value


########################################################################################
# default pipeline loading data from tar files into a tuple (sample_id, x, y)

Sample = collections.namedtuple("Sample", ["sample_id", "x", "y"])


def construct_sample_datapipe(
    shard_folder: pathlib.Path,
    num_workers: int,
    buffer_size: int = 0,
    shuffle_shards_on_epoch: bool = False,
) -> IterDataPipe[Sample]:
    # list all shards
    shard_list = [str(f) for f in shard_folder.glob("shard-*.tar")]

    if len(shard_list) == 0:
        raise ValueError(f"unable to find any shards in {shard_folder}")

    # stream of strings representing each shard
    dp = FileLister(shard_list)

    # shuffle the stream so order of shards in epoch differs
    if shuffle_shards_on_epoch:
        dp = Shuffler(dp, buffer_size=len(shard_list))

    # make sure each worker receives the same number of shards
    if num_workers > 0:
        if len(shard_list) < num_workers:
            raise ValueError(f"{num_workers=} cannot be smaller than {len(shard_list)=}")

        dp = Header(dp, limit=len(shard_list) // num_workers)

        # each worker only sees 1/n elements
        dp = ShardingFilter(dp)

    # map strings of paths to file handles
    dp = FileOpener(dp, mode="b")

    # expand each file handle to a stream of all files in the tar
    dp = TarArchiveLoader(dp, mode="r")

    # decode each file in the tar to the expected python dataformat
    dp = Mapper(dp, decode)

    # each file in the tar is expected to have the format `{key}.{ext}
    # this groups all files with the same key into one dictionary
    dp = WebDataset(dp)

    # transform the dictionaries into tuple (sample_id, x, y)
    dp = Mapper(dp, map_dict_to_tuple)

    # buffer tuples to increase variability
    if buffer_size > 0:
        dp = Shuffler(dp, buffer_size=buffer_size)
    return dp


def map_dict_to_tuple(x: Dict) -> Sample:
    sample_id = x[".json"]["sample_id"]
    wav = x[".wav"]

    class_idx = x[".json"]["class_idx"]
    if class_idx is None:
        gt = None
    else:
        gt = t.tensor(x[".json"]["class_idx"], dtype=t.int64)

    return Sample(sample_id, wav, gt)


########################################################################################
# useful transformation on a stream of sample objects


def _chunk_sample(sample: Sample, num_frames: int):
    sample_id, x, y = sample

    sample_length = x.shape[0]
    start_idx = t.randint(low=0, high=sample_length - num_frames - 1, size=())
    end_idx = start_idx + num_frames

    assert len(x.shape) == 1  # before e.g. mfcc transformation
    x = x[start_idx:end_idx]

    return Sample(sample_id, x, y)


def pipe_chunk_sample(
    dp: IterDataPipe[Sample], num_frames: int
) -> IterDataPipe[Sample]:
    return Mapper(dp, functools.partial(_chunk_sample, num_frames=num_frames))


def _mfcc_sample(sample: Sample, mfcc: torchaudio.transforms.MFCC):
    sample_id, x, y = sample

    # we go from shape [num_frames] to [num_mel_coeff, num_frames/window_size]
    x = mfcc(x)

    return Sample(sample_id, x, y)


def pipe_mfcc(dp: IterDataPipe[Sample], n_mfcc: int) -> IterDataPipe[Sample]:
    mfcc = torchaudio.transforms.MFCC(n_mfcc=n_mfcc)

    return Mapper(dp, functools.partial(_mfcc_sample, mfcc=mfcc))


def _batch_samples(samples: List[Sample]):
    assert len(samples) > 0

    shapes = {s.x.shape for s in samples}
    assert len(shapes) == 1  # all samples have same shape

    x = torch.utils.data.default_collate([s.x for s in samples])

    if samples[0].y is not None:
        y = torch.utils.data.default_collate([s.y for s in samples])
    else:
        y = None

    sample_id = [s.sample_id for s in samples]

    return Sample(sample_id, x, y)


def pipe_batch_samples(
    dp: IterDataPipe[Sample], batch_size: int, drop_last: bool = False
) -> IterDataPipe[Sample]:
    return Batcher(dp, batch_size, drop_last=drop_last, wrapper_class=_batch_samples)


########################################################################################
# useful for debugging a datapipe


def _print_sample(dp):
    for sample in dp:
        sample_id, x, y = sample
        print(f"{sample_id=}\n")

        print(f"{x.shape=}")
        print(f"{x.dtype=}\n")

        print(y)
        print(f"{y.shape=}")
        print(f"{y.dtype=}\n")
        break

def debug_an():
    shard_path = pathlib.Path(
        "/home/anilsson/mlip/tiny-voxceleb-skeleton-2023/data/tiny-voxceleb-shards/train"
    )

    n_mfcc = 40

    print("### construct_sample_datapipe ###")
    dp = construct_sample_datapipe(shard_path, num_workers=0)
    _print_sample(dp)

    print("### pipe_chunk_sample ###")
    dp = pipe_chunk_sample(dp, 16_000 * 3)  # 3 seconds

    _print_sample(dp)



def _debug():
    shard_path = pathlib.Path(
        "/home/lnguyen/mlip/tiny-voxceleb-skeleton-2023/data/tiny-voxceleb-shards/train"
    )

    n_mfcc = 40

    print("### construct_sample_datapipe ###")
    dp = construct_sample_datapipe(shard_path, num_workers=0)
    _print_sample(dp)

    print("### pipe_chunk_sample ###")
    dp = pipe_chunk_sample(dp, 16_000 * 3)  # 3 seconds

    _print_sample(dp)

    

    print("### pipe_mfcc ###")
    dp = pipe_mfcc(dp, n_mfcc)
    _print_sample(dp)

    print("### pipe_batch_samples ###")
    dp = pipe_batch_samples(dp, 8)
    _print_sample(dp)






if __name__ == "__main__":
    #_debug()
    debug_an()
