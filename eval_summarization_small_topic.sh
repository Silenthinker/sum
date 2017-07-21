export PYTHONIOENCODING=UTF-8
export DATA_PATH="$(pwd)/data/giga_small_small"

export VOCAB_SOURCE=${DATA_PATH}/vocab.50k.art
export VOCAB_TARGET=${DATA_PATH}/vocab.50k.sum
export TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.art
export TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.sum
export DEV_SOURCES=${DATA_PATH}/valid.tok.clean.art
export DEV_TARGETS=${DATA_PATH}/valid.tok.clean.sum
export TEST_SOURCES=${DATA_PATH}/test.tok.clean.art
export TEST_TARGETS=${DATA_PATH}/test.tok.clean.sum
export TOPIC_MODEL="$(pwd)"/data/giga_lda_model0716

export MODEL_DIR="$(pwd)/sum_conv_seq2seq_topic2"
export PRED_DIR=${MODEL_DIR}/pred

mkdir -p ${PRED_DIR}

###with greedy search
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --model_params "
    inference.beam_search.beam_width: 1 
    decoder.class: seq2seq.decoders.ConvDecoderFairseqTopic" \
  --input_pipeline "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TEST_SOURCES" \
  > ${PRED_DIR}/predictions.txt

: <<END
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


./bin/tools/multi-bleu.perl ${TEST_TARGETS} < ${PRED_DIR}/predictions.txt
END
