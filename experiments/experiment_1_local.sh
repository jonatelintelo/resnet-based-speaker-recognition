#!/usr/bin/env bash
### notes
# this experiment is meant to try out training the default model

# easy access to file locations
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR=$PROJECT_DIR/data

# location of repository and data
project_dir=$PROJECT_DIR
shard_folder=$DATA_DIR/tiny-voxceleb-shards
val_trials_path=$DATA_DIR/tiny-voxceleb/val_trials.txt
dev_trials_path=$DATA_DIR/tiny-voxceleb/dev_trials.txt

# hyperparameters for optimization
batch_size=128
learning_rate=3e-3
num_epochs=30
num_workers=0

# hyperparameters related to data pre-processing and network architecture
audio_length_num_frames=48000
n_mfcc=40
embedding_size=128

# execute train CLI
source "$project_dir"/venv/bin/activate
python "$project_dir"/cli_train.py \
  --shard_folder "$shard_folder" \
  --val_trials_path "$val_trials_path" \
  --dev_trials_path "$dev_trials_path" \
  --batch_size $batch_size \
  --audio_length_num_frames $audio_length_num_frames \
  --n_mfcc $n_mfcc \
  --embedding_size $embedding_size \
  --learning_rate $learning_rate \
  --epochs $num_epochs \
  --num_workers $num_workers