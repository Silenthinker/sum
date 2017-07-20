export PYTHONIOENCODING=UTF-8
export DATA_PATH="$(pwd)/data/iwslt14.tokenized.de-en"

export VOCAB_SOURCE=${DATA_PATH}/vocab.bpe.32000
export VOCAB_TARGET=${DATA_PATH}/vocab.bpe.32000
export TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.bpe.32000.de
export TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.bpe.32000.en
export DEV_SOURCES=${DATA_PATH}/valid.tok.clean.bpe.32000.de
export DEV_TARGETS=${DATA_PATH}/valid.tok.clean.bpe.32000.en
export TEST_SOURCES=${DATA_PATH}/test.tok.clean.bpe.32000.de
export TEST_TARGETS=${DATA_PATH}/test.tok.clean.bpe.32000.en

export MODEL_DIR="$(pwd)/iwslt_conv_seq2seq"
export PRED_DIR=${MODEL_DIR}/pred

mkdir -p ${PRED_DIR}

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
  > ${PRED_DIR}/predictions.txt

: <<'END'
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
  > ${PRED_DIR}/predictions.txt
END
./bin/tools/multi-bleu.perl ${TEST_TARGETS} < ${PRED_DIR}/predictions.txt
