export PYTHONIOENCODING=UTF-8

DATA_PATH="$(pwd)/data/giga_small_small"
TEST_SOURCES=${DATA_PATH}
MODEL_DIR="$(pwd)/sum_conv_seq2seq_rl"
###DATA_PATH=$1
###TEST_SOURCES=$2
###MODEL_DIR=$3

export VOCAB_SOURCE=${DATA_PATH}/vocab.50k.art
export VOCAB_TARGET=${DATA_PATH}/vocab.50k.sum
export TRAIN_SOURCES=${DATA_PATH}/train.art.small
export TRAIN_TARGETS=${DATA_PATH}/train.sum.small
export DEV_SOURCES=${DATA_PATH}/valid.art.small
export DEV_TARGETS=${DATA_PATH}/valid.sum.small
export TEST_SOURCES=${DATA_PATH}/test.art.small
export TEST_TARGETS=${DATA_PATH}/test.sum.small
export TOPIC_MODEL="$(pwd)"/data/giga_lda_model0716_

##export PRED_DIR=${DATA_PATH}/summary
export PRED_DIR=${MODEL_DIR}/summary
export GREEDY_DIR=${PRED_DIR}/greedy
export BEAM_DIR=${PRED_DIR}/beam
mkdir -p ${PRED_DIR}
mkdir -p ${GREEDY_DIR}
mkdir -p ${BEAM_DIR}


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
    | sed 's/@@ //g'> ${PRED_DIR}/summaryA.txt
cp ${PRED_DIR}/summaryA.txt ${GREEDY_DIR}/summaryA.txt
rm ${PRED_DIR}/summaryA.txt

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
cp ${PRED_DIR}/summaryA.txt ${BEAM_DIR}/summaryA.txt
rm ${PRED_DIR}/summaryA.txt


echo "Greedy result (origin):" 
python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir $DATA_PATH/reference/origin -sum_dir ${GREEDY_DIR}
echo "Beam result (origin):"
python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir $DATA_PATH/reference/origin -sum_dir ${BEAM_DIR}

echo "Greedy result (tok.clean):"
python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir $DATA_PATH/reference/tok.clean -sum_dir ${GREEDY_DIR}
echo "Beam result (tok.clean):"
python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir $DATA_PATH/reference/tok.clean -sum_dir ${BEAM_DIR}

"""
(tensorflow) nlp@nlp-workstation:/nlp/lilianwang/conv_seq2seq_master$ python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir ./sum_conv_seq2seq_rl/reference -sum_dir ./sum_conv_seq2seq_rl/summary/greedy/
{'ROUGE-1': 0.82474, 'ROUGE-2': 0.69474, 'ROUGE-L': 0.82474, 'ROUGE-SU4': 0.72}
(tensorflow) nlp@nlp-workstation:/nlp/lilianwang/conv_seq2seq_master$ python seq2seq/metrics/pythonrouge/rouge_scorer.py -ref_dir ./sum_conv_seq2seq_rl/reference -sum_dir ./sum_conv_seq2seq_rl/summary/beam/
{'ROUGE-1': 0.76288, 'ROUGE-2': 0.61053, 'ROUGE-L': 0.76288, 'ROUGE-SU4': 0.62545}
"""
