#!/bin/bash
### Usage ###
# This script consists in downsampling given files
# Make sure that the user has permission
# If permission denied
# Type:
# sudo chmod 775
# then simply type:
# ./downsampling.sh

n_train=1
n_valid=10000
# n_test=1951

seed=10

DATA_PATH="$(pwd)/data/sum_data"

TRAIN_SOURCES=train.tok.clean.bpe.32000.art
TRAIN_TARGETS=train.tok.clean.bpe.32000.sum
DEV_SOURCES=valid.tok.clean.bpe.32000.art
DEV_TARGETS=valid.tok.clean.bpe.32000.sum

OUTPUT_PATH="$(pwd)/data/giga_small"

get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}

shuf -n $n_train --random-source=<(get_seeded_random $seed) ${DATA_PATH}/$TRAIN_SOURCES > $OUTPUT_PATH/$TRAIN_SOURCES
shuf -n $n_train --random-source=<(get_seeded_random $seed) ${DATA_PATH}/$TRAIN_TARGETS > $OUTPUT_PATH/$TRAIN_TARGETS
shuf -n $n_valid --random-source=<(get_seeded_random $seed) ${DATA_PATH}/$DEV_SOURCES > $OUTPUT_PATH/$DEV_SOURCES
shuf -n $n_valid --random-source=<(get_seeded_random $seed) ${DATA_PATH}/$DEV_TARGETS > $OUTPUT_PATH/$DEV_TARGETS