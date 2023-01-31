#! /usr/bin/env bash

srun --gres=gpu:1 --time=4:00:00 --mem=10G --cpus-per-task=5  -p csedu --pty bash
