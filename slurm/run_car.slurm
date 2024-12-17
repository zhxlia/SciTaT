MODEL=Llama-3.1-Instruct
SCALE=8B
PAPER_RETRIEVE=short_context
SHOT_N=0


export VLLM_WORKER_MULTIPROC_METHOD=spawn
DATA_PATH=./baseline/result/$MODEL-$SCALE/CaR/$PAPER_RETRIEVE
[ ! -d "$DATA_PATH" ] && mkdir -p "$DATA_PATH"
python3 ./baseline/calculator.py \
    --model ./model/$MODEL/$SCALE \
    --config_file ./config/$MODEL.json \
    --questions_file ./data/SciTQA/test.json \
    --language zh \
    --retrieve $PAPER_RETRIEVE \
    --shot_num $SHOT_N \
    --dump_file $DATA_PATH/calculate.json


python3 ./baseline/reasoner.py \
    --model ./model/$MODEL/$SCALE \
    --config_file ./config/$MODEL.json \
    --questions_file $DATA_PATH/calculate.json \
    --language en \
    --retrieve $PAPER_RETRIEVE \
    --reasoning_type $REASONING \
    --shot_num $SHOT_N \
    --dump_file $DATA_PATH/reason.json \
    # --papers_file ./data/text/main.w.table.json \
    # --tables_file ./data/table/main.json \