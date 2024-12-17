QUES_FILE=./baseline/result/Llama-3.1-Instruct-8B/CaR/short-context/reason.json
echo $QUES_FILE
python3 ./baseline/evaluate.py \
    --questions_file $QUES_FILE \
    --gold_file ./data/SciTQA/test.json \
    --language en \
    --dump_file $QUES_FILE
    # --reasoning_type CoT \
    # --if_process true