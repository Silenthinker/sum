#!/bin/bash
### Usage ###
# This script consists in downsampling given files
# Make sure that the user has permission
# If permission denied
# Type:
# sudo chmod 775
# then simply type:
# ./downsampling.sh

n_train=100000
n_valid=10000
# n_test=1951

seed=5

DATA_PATH="$(pwd)/data/sum_data"

TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.bpe.32000.art
TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.bpe.32000.sum
DEV_SOURCES=${DATA_PATH}/valid.tok.clean.bpe.32000.art
DEV_TARGETS=${DATA_PATH}/valid.tok.clean.bpe.32000.sum



get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

shuf -n $n_train --random-source=<(get_seeded_random $seed) $TRAIN_SOURCES > $TRAIN_SOURCES.small
shuf -n $n_train --random-source=<(get_seeded_random $seed) $TRAIN_TARGETS > $TRAIN_TARGETS.small
shuf -n $n_valid --random-source=<(get_seeded_random $seed) $DEV_SOURCES > $DEV_SOURCES.small
shuf -n $n_valid --random-source=<(get_seeded_random $seed) $DEV_TARGETS > $DEV_TARGETS.small