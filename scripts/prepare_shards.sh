#! /usr/bin/env bash
set -e

if [[ -z "$DATA_FOLDER" ]]; then
  echo "Please set the DATA_FOLDER environment variable before calling this script."
  exit 1
fi

# set variable to path where this script is
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1

# write shards for each subset of the data
python write_tar_shards.py  --in "$DATA_FOLDER"/tiny-voxceleb/train --meta "$DATA_FOLDER"/tiny-voxceleb/tiny_meta.csv --out "$DATA_FOLDER"/tiny-voxceleb-shards/train
python write_tar_shards.py  --in "$DATA_FOLDER"/tiny-voxceleb/val   --meta "$DATA_FOLDER"/tiny-voxceleb/tiny_meta.csv --out "$DATA_FOLDER"/tiny-voxceleb-shards/val
python write_tar_shards.py  --in "$DATA_FOLDER"/tiny-voxceleb/dev   --meta "$DATA_FOLDER"/tiny-voxceleb/tiny_meta.csv --out "$DATA_FOLDER"/tiny-voxceleb-shards/dev
python write_tar_shards.py  --in "$DATA_FOLDER"/tiny-voxceleb/eval  --meta "$DATA_FOLDER"/tiny-voxceleb/tiny_meta.csv --out "$DATA_FOLDER"/tiny-voxceleb-shards/eval
