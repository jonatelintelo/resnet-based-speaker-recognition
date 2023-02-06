#! /usr/bin/env bash
set -e

# check whether DATA_FOLDER and SCIENCE_USERNAME have been set
if [[ -z "$DATA_FOLDER" ]]; then
  echo "Please set the DATA_FOLDER environment variable before calling this script"
  exit 1
fi
if [[ -z "$SCIENCE_USERNAME" ]]; then
  echo "Please set the SCIENCE_USERNAME environment variable before calling this script"
  exit 1
fi

# make sure the data folder exists
mkdir -p "$DATA_FOLDER"

# rsync remote logs
rsync -azP "$SCIENCE_USERNAME"@slurm22.science.ru.nl:/ceph/csedu-scratch/course/IMC030_MLIP/users/"$SCIENCE_USERNAME"/ "$DATA_FOLDER"