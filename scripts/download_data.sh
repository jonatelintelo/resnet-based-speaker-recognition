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

# rsync data from cluster to the local data folder
USERNAME=your_username
rsync -P "$SCIENCE_USERNAME"@cn99.science.ru.nl:/ceph/csedu-scratch/course/IMC030_MLIP/data/data.zip "$DATA_FOLDER"/data.zip

# now you can unzip, by doing:
# unzip "$DATA_FOLDER"/data.zip $DATA_FOLDER
