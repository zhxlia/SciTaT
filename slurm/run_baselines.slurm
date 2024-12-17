MODEL=Llama-3.1-Instruct
SCALE=70B
PAPER_RETRIEVE=short_context
SHOT_N=0
REASONING=CoT
LANGUAGE=en


export VLLM_WORKER_MULTIPROC_METHOD=spawn
DATA_PATH=./baseline/result/$MODEL-$SCALE/$REASONING/$PAPER_RETRIEVE
[ ! -d "$DATA_PATH" ] && mkdir -p "$DATA_PATH"
python3 ./baseline/baseline.py \
    --model ./model/$MODEL/$SCALE \
    --config_file ./config/$MODEL.json \
    --questions_file ./data/SciTQA/test.json \
    --language $LANGUAGE \
    --retrieve $PAPER_RETRIEVE \
    --reasoning_type $REASONING \
    --shot_num $SHOT_N \
    --dump_file $DATA_PATH/test.json \
    # --papers_file ./data/text/main.w.table.json \
    # --tables_file ./data/table/main.json \