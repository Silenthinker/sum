#! /usr/bin/env bash

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

# BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../.." && pwd )"
BASE_DIR="$(pwd)"

OUTPUT_DIR="${BASE_DIR}/data/sum_data"
echo "Writing to ${OUTPUT_DIR}. To change this, set the OUTPUT_DIR environment variable."

# Clone Moses
if [ ! -d "${OUTPUT_DIR}/mosesdecoder" ]; then
  echo "Cloning moses for data processing"
  git clone https://github.com/moses-smt/mosesdecoder.git "${OUTPUT_DIR}/mosesdecoder"
fi

# Tokenize data
for f in ${OUTPUT_DIR}/*.sum; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.sum
done

for f in ${OUTPUT_DIR}/*.art; do
  echo "Tokenizing $f..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -q -l en -threads 8 < $f > ${f%.*}.tok.art
done

# Clean all corpora
for f in ${OUTPUT_DIR}/*.art; do
  fbase=${f%.*}
  echo "Cleaning ${fbase}..."
  ${OUTPUT_DIR}/mosesdecoder/scripts/training/clean-corpus-n.perl $fbase 'sum' 'art' "${fbase}.clean" 1 80
done


# Create vocabulary for non-tokenized art data
$BASE_DIR/bin/tools/generate_vocab.py \
   --max_vocab_size 50000 \
  < ${OUTPUT_DIR}/train.art \
  > ${OUTPUT_DIR}/vocab.50k.art \

# Create vocabulary for non-tokenized sum data
$BASE_DIR/bin/tools/generate_vocab.py \
  --max_vocab_size 50000 \
  < ${OUTPUT_DIR}/train.sum \
  > ${OUTPUT_DIR}/vocab.50k.sum \

# Create character vocabulary (on tokenized data)
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.tok.clean.art \
  > ${OUTPUT_DIR}/vocab.tok.char.art
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.tok.clean.sum \
  > ${OUTPUT_DIR}/vocab.tok.char.sum

# Create character vocabulary (on non-tokenized data)
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.clean.art \
  > ${OUTPUT_DIR}/vocab.char.art
${BASE_DIR}/bin/tools/generate_vocab.py --delimiter "" \
  < ${OUTPUT_DIR}/train.clean.sum \
  > ${OUTPUT_DIR}/vocab.char.sum

# Create vocabulary for art data
$BASE_DIR/bin/tools/generate_vocab.py \
   --max_vocab_size 50000 \
  < ${OUTPUT_DIR}/train.tok.clean.art \
  > ${OUTPUT_DIR}/vocab.tok.clean.50k.art \

# Create vocabulary for sum data
$BASE_DIR/bin/tools/generate_vocab.py \
  --max_vocab_size 50000 \
  < ${OUTPUT_DIR}/train.tok.clean.sum \
  > ${OUTPUT_DIR}/vocab.tok.clean.50k.sum \

# Generate Subword Units (BPE)
# Clone Subword NMT
if [ ! -d "${OUTPUT_DIR}/subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git "${OUTPUT_DIR}/subword-nmt"
fi

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "${OUTPUT_DIR}/train.tok.clean.sum" "${OUTPUT_DIR}/train.tok.clean.art" | \
    ${OUTPUT_DIR}/subword-nmt/learn_bpe.py -s $merge_ops > "${OUTPUT_DIR}/bpe.${merge_ops}"

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in 'art' 'sum'; do
    for f in ${OUTPUT_DIR}/*.tok.${lang} ${OUTPUT_DIR}/*.tok.clean.${lang}; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${OUTPUT_DIR}/subword-nmt/apply_bpe.py -c "${OUTPUT_DIR}/bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE
  cat "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.art" "${OUTPUT_DIR}/train.tok.clean.bpe.${merge_ops}.sum" | \
    ${OUTPUT_DIR}/subword-nmt/get_vocab.py | cut -f1 -d ' ' > "${OUTPUT_DIR}/vocab.bpe.${merge_ops}"

done

echo "All done."
