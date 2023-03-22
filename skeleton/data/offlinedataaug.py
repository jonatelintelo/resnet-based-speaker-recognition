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
import time
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

import librosa
import soundfile
import os.path
import os
import csv
import numpy as np 

########################################################################################
#Timer 

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time:.5f} seconds")
        return result
    return wrapper

########################################################################################

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

@timer #written in log everytime it is called, for debugging 
def reverb_aug(data,sample_rate): #rir =Room Impulse Respons # this is very time consuming. Maybe do on batch instead. 
	rir_raw = data
	rir = rir_raw[:, int(sample_rate * 1.01) : 	int(sample_rate * 1.3)]
	rir = rir / torch.norm(rir, p=2)
	RIR = torch.flip(rir, [1])
	pad_data = torch.nn.functional.pad(data, (RIR.shape[1] - 1, 0)) #adds 0 before and after timeseries
	aug= torch.nn.functional.conv1d(pad_data.unsqueeze(0), RIR.unsqueeze(0))[0] # (minibatch,in_channels,iW), unsqueeze adds dim so it works with conv
	return aug
 

def pitch_aug(data, samplingrate, semitones): #only +- 4 or 5 semitones
    aug= librosa.effects.pitch_shift(y=data.numpy(),sr= samplingrate,n_steps= semitones)
    return torch.tensor(aug)


def random_gain_aug(data, minimum=0.1, maximum=0.12): #change the percieved loudness of the waveform
    gain = random.uniform(minimum, maximum) 
    return data * gain #scale but in amplitude 



def decode_wav(value: StreamWrapper, choice) -> t.Tensor:
   # assert isinstance(value, StreamWrapper) #ehh is this needed?

    semitones=4

    value, sample_rate = torchaudio.load(value)

    if choice == 'inject_noise':
        value = inject_noise(value, 0.01)
    elif choice == 'rd_speed_change':
        value = random_speed_change(value, sample_rate)
    elif choice == 'rand_gain':
        value= random_gain_aug(value, minimum=0.1, maximum=0.12)
    elif choice == 'reverb':
        value= reverb_aug(value,sample_rate)
    elif choice == 'pitch':
        value= pitch_aug(value, sample_rate, semitones)

    assert sample_rate == 16_000

    # make sure that audio has 1 dimension # no two 
    #value = torch.squeeze(value)

   # print("end decode")    
    return value, sample_rate

# calling to store shorter bits 
def _chunk_sample(sample, num_frames=3):
    x = sample

    sample_length = x.shape[0]
    start_idx = t.randint(low=0, high=sample_length - num_frames - 1, size=())
    end_idx = start_idx + num_frames

    assert len(x.shape) == 1  # before e.g. mfcc transformation
    x = x[start_idx:end_idx]

    return x



def aug_(path = "./data/tiny-voxceleb/train",new_path ="./skeleton/data/aug"):
    print(path,new_path)

    c_list = ['inject_noise', 'rd_speed_change','rand_gain', 'reverb','pitch', 'none']

    # only needs to be used once so excuse the inefficiency 
    #print("starting forloop")
    for choice in c_list: # loop through augmentation options
        #print("CHOICE:", choice)
        for wav in os.listdir(path): #enter wavdirectory
         
            wav_path = os.path.join(path, wav)  # relative, therefore join with prev
            for ids in os.listdir(wav_path): #loop through people
                id_path = os.path.join(wav_path, ids)  # relative, therefore join with prev

                for sample in os.listdir(id_path): #loop through samples
                    sample_path = os.path.join(id_path, sample)
                    i=1
                    subdir_path=f"{new_path}/wav/{ids}_{choice}/{sample}_{choice}"
                    # print(sample,choice)
                    os.makedirs(subdir_path) #creating all subdirectories

                    for sample_wav in os.listdir(sample_path): #loop through each file 

         #               print("SAMPLE:",i,"/",len(os.listdir(sample_path)))

                        file_path = os.path.join(sample_path, sample_wav)

                        aug_file_tensor,sample_rate= decode_wav(file_path,choice) # returns augmented tensor and sample rate  
                        
                        #toerchaudio instead of soundfile 
                        torchaudio.save(f"{subdir_path}/{sample_wav.strip('.wav')}_{choice}.wav", aug_file_tensor, sample_rate) #write each 
                        # same directory structiure, new choice-specific lables 
                    i+=1


    ## Create tar files seperately using given script 


def make_metafile():

    c_list = ['inject_noise', 'rd_speed_change','rand_gain', 'reverb','pitch', 'none'] #none files are actually just without "_choice", did rm after

    #hard coded paths bc wont need ágain, hopefully 
    meta_path= "skeleton/data/aug/tiny_meta_aug.csv" #could not place in tiny-voxceleb
    org_meta_path="data/tiny-voxceleb/tiny_meta.csv" #original

    # extracting IDs from metafile
    org_meta = open(org_meta_path,"r")  #reading from
    ids=[]
    reader=csv.reader(org_meta)
    for row in reader:
        ids.append(row)
    org_meta.close()

    # building new metafile with augmentation lables
    new_meta = open(meta_path,"w")      #writing to 
    writer=csv.writer(new_meta)
    for choice in c_list:
        ids_copy=[i.copy() for i in ids]
        for person in ids_copy[1:]:
            person[0]=person[0]+"_"+choice
            writer.writerow(person)
    new_meta.close()

def debug_an():


    print("running aug")
    aug_("/home/alma/Documents/Radboud/MLiP/tiny-voxceleb-skeleton-2023/data/tiny-voxceleb/train","./skeleton/data/aug/illustration_augmentation")



if __name__ == "__main__":
    #_debug()
    debug_an()
