export PYTHONIOENCODING=UTF-8

DATA_PATH="$(pwd)/sum_data"
TEST_SOURCES=${DATA_PATH}
MODEL_DIR="$(pwd)/sum_conv_seq2seq_rl"
###DATA_PATH=$1
###TEST_SOURCES=$2
###MODEL_DIR=$3

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
