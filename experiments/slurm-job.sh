#!/bin/bash

## Bail out on errors
set -e

echo "Start of my script"

echo "The time is $(date)"

echo "I am running on machine $(hostname)"

echo "I am running this from the folder $(pwd)"

echo "I know the following environment variables:"
printenv

echo "This is the ouput of nvidia-smi:"
nvidia-smi

echo "Pretending to be busy for a while"
sleep 10

echo "This is enough, the time is now $(date)"

exit 0

