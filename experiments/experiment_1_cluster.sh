#!/usr/bin/env bash
#SBATCH --partition=csedu
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=6
#SBATCH --time=6:00:00
#SBATCH --output=./logs/slurm/%J.out
#SBATCH --error=./logs/slurm/%J.err
#only use this if you want to send the mail to another team member #SBATCH --mail-user=teammember
#SBATCH --mail-type=BEGIN,END,FAIL

#### notes
# this experiment is meant to try out training the default model

# location of repository and data
project_dir=.
shard_folder=./data/tiny-voxceleb-shards/
val_trials_path=./data/tiny-voxceleb/val_trials.txt
dev_trials_path=./data/tiny-voxceleb/dev_trials.txt

# hyperparameters for optimization
batch_size=128
learning_rate=3e-3
num_epochs=30
num_workers=5

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