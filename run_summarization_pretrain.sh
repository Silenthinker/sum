export PYTHONIOENCODING=UTF-8
export DATA_PATH="$(pwd)/data/giga_small_small"

export VOCAB_SOURCE=${DATA_PATH}/vocab.50k.art
export VOCAB_TARGET=${DATA_PATH}/vocab.50k.sum
export TRAIN_SOURCES=${DATA_PATH}/train.art.small
export TRAIN_TARGETS=${DATA_PATH}/train.sum.small
export DEV_SOURCES=${DATA_PATH}/valid.art.small
export DEV_TARGETS=${DATA_PATH}/valid.sum.small
export TEST_SOURCES=${DATA_PATH}/test.art.small
export TEST_TARGETS=${DATA_PATH}/test.sum.small
export TOPIC_MODEL="$(pwd)"/data/giga_lda_model0716_

export TRAIN_STEPS=1000000

export MODEL_DIR="$(pwd)/sum_conv_seq2seq_rl"
mkdir -p $MODEL_DIR

python -m bin.train \
  --config_paths="
      ./example_configs/conv_seq2seq_pretrain.yml,
      ./example_configs/train_seq2seq.yml,
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
  --eval_every_n_steps 100000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR
