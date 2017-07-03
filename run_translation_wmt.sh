export PYTHONIOENCODING=UTF-8
export DATA_PATH="$(pwd)/data/wmt16_en_de"

export VOCAB_SOURCE=${DATA_PATH}/vocab.bpe.32000
export VOCAB_TARGET=${DATA_PATH}/vocab.bpe.32000
export TRAIN_SOURCES=${DATA_PATH}/train.tok.clean.bpe.32000.de
export TRAIN_TARGETS=${DATA_PATH}/train.tok.clean.bpe.32000.en
export DEV_SOURCES=${DATA_PATH}/newstest2016.tok.bpe.32000.de
export DEV_TARGETS=${DATA_PATH}/newstest2016.tok.bpe.32000.en
export TEST_SOURCES=${DATA_PATH}/newstest2013.tok.bpe.32000.de
export TEST_TARGETS=${DATA_PATH}/newstest2013.tok.bpe.32000.en

export TRAIN_STEPS=1000000

export MODEL_DIR="$(pwd)/nmt_conv_seq2seq"
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET" \
  --input_pipeline_train "
    class: ParallelTextInputPipelineFairseq
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS" \
  --input_pipeline_dev "
    class: ParallelTextInputPipelineFairseq
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS" \
  --batch_size 32 \
  --eval_every_n_steps 5000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
