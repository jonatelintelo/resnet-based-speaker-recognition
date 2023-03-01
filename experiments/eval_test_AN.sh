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

# location of repository and data
project_dir=.
shard_folder=./data/tiny-voxceleb-shards/
val_trials_path=./data/tiny-voxceleb/val_trials.txt
dev_trials_path=./data/tiny-voxceleb/dev_trials.txt

# execute train CLI


source "$project_dir"/venv/bin/activate
./cli_evaluate.py \
logs/lightning_logs/version_2023_02_24___00_02_28___job_id_15379/checkpoints/epoch_0027.step_000001092.val-eer_0.1586.best.ckpt \
data/tiny-voxceleb-shards/dev/,data/tiny-voxceleb-shards/eval \
data/tiny-voxceleb/dev_trials.txt,data/tiny-voxceleb/eval_trials_no_gt.txt \
scores.txt
