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

export TRAIN_STEPS=100000

export MODEL_DIR="$(pwd)/sum_conv_seq2seq_topic2"
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq_topic.yml,
      ./example_configs/train_seq2seq_sum.yml,
      ./example_configs/text_metrics_bpe.yml" \
  --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET
      topic_model: $TOPIC_MODEL" \
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
