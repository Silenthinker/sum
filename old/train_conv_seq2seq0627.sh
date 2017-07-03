export PYTHONIOENCODING=UTF-8

export DATA_PATH=/nlp/lilianwang/conv_seq2seq_master/sum_data

export VOCAB_SOURCE=${DATA_PATH}/doc_dict.txt
export VOCAB_TARGET=${DATA_PATH}/sum_dict.txt
export TRAIN_SOURCES=${DATA_PATH}/train.article.txt
export TRAIN_TARGETS=${DATA_PATH}/train.title.txt
export DEV_SOURCES=${DATA_PATH}/valid.article.filter.txt
export DEV_TARGETS=${DATA_PATH}/valid.title.filter.txt
export TEST_SOURCES=${DATA_PATH}/input.txt
export TEST_TARGETS=${DATA_PATH}/task1_ref0.txt

export TRAIN_STEPS=1000000


##export MODEL_DIR=${TMPDIR:-/tmp}/nmt_conv_seq2seq
export MODEL_DIR=/nlp/lilianwang/conv_seq2seq_master/sum_data/conv_seq2seq_giga
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
  --batch_size 64 \
  --eval_every_n_steps 5000 \
  --train_steps $TRAIN_STEPS \
  --output_dir $MODEL_DIR


