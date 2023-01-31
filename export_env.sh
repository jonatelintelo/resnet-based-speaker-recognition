#! /usr/bin/env bash

# You can use this script to load all the environment variables in the `.env` file.
# Make sure your current working directory is in the same location as the .env file.
#
# Usage:
#  . ./export-env.sh

eval "$(
  cat .env | awk '!/^\s*#/' | awk '!/^\s*$/' | while IFS='' read -r line; do
    key=$(echo "$line" | cut -d '=' -f 1)
    value=$(echo "$line" | cut -d '=' -f 2-)
    echo "export $key=\"$value\""
  done
)"