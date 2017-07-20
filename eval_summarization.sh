export PYTHONIOENCODING=UTF-8
export DATA_PATH="$(pwd)/data/giga_small"

export VOCAB_SOURCE=${DATA_PATH}/vocab.bpe.32000
export VOCAB_TARGET=${DATA_PATH}/vocab.bpe.32000
export TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.bpe.32000.art
export TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.bpe.32000.sum
export DEV_SOURCES=${DATA_PATH}/valid.tok.clean.bpe.32000.art
export DEV_TARGETS=${DATA_PATH}/valid.tok.clean.bpe.32000.sum
export TEST_SOURCES=${DATA_PATH}/test.tok.clean.bpe.32000.art
export TEST_TARGETS=${DATA_PATH}/test.tok.clean.bpe.32000.sum

export MODEL_DIR="$(pwd)/sum_conv_seq2seq"
export PRED_DIR=${DATA_PATH}/summary

mkdir -p ${PRED_DIR}

: <<'END'
echo "Greedy search..."
###with greedy search
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseq" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/summaryA.txt
END

echo "Beam search..."
###with beam search
python -m bin.infer \
  --tasks "
    - class: DecodeText
    - class: DumpBeams
      params:
        file: ${PRED_DIR}/beams.npz" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 5 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqBS" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  | sed 's/@@ //g'> ${PRED_DIR}/summaryA.txt

# ./bin/tools/multi-bleu.perl ${TEST_TARGETS} < ${PRED_DIR}/summaryA.txt

echo "Prediction Done!"

python3 seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir data/giga_small/reference/ -sum_dir data/giga_small/summary/
