import os
import re
import sys
import ast
import json
import random
import argparse
import numpy as np

sys.path.append('.')
a_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(a_directory)

from tqdm import tqdm
from copy import deepcopy
from bert_score import score
from transformers import set_seed
from collections import defaultdict
from argparse import ArgumentParser
from rouge_score import rouge_scorer
from nltk.translate import meteor_score
from typing import List, Dict, Any, Union, Tuple

random.seed(42)



def compute_meteor(predictions, references, alpha=0.9, beta=3, gamma=0.5):
    scores = [meteor_score.single_meteor_score(ref, pred, alpha=alpha, beta=beta, gamma=gamma)
                  for ref, pred in zip(references, predictions)]

    return {"meteor": np.mean(scores)}

def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for ref, pred in zip(references, predictions)]
    return {"rougeL": np.mean(rouge_scores)}


def json_to_txt(data: List[str], output_file: str):
    with open(output_file, 'w', encoding='utf-8') as f:
        for d in data:
            d = d.replace("\n", " ").replace("\r", " ").replace("\f", " ")
            f.write(d + "\n")
    return data



if __name__ == '__main__':
    from utils.tatqa_metric import TaTQAEmAndF1
    from baseline.baseline import process_output_to_answer


    parser = argparse.ArgumentParser()
    parser.add_argument("--questions_file", type=str, help="questions file")
    parser.add_argument("--language", type=str, default="en", help="language", choices=["en", "zh"])
    parser.add_argument("--gold_file", type=str, help="dump path")
    parser.add_argument("--data_size", type=int, help="data size")
    parser.add_argument("--if_process", type=bool, default=False, help="if process")
    parser.add_argument("--reasoning_type", type=str, help="reasoning type")
    parser.add_argument("--dump_file", type=str, help="dump file")
    parser.add_argument("--random_seed", type=int, default=42, help="random seed")
    args = parser.parse_args()
    set_seed(args.random_seed)

    questions_file = args.questions_file
    data_size = args.data_size

    
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        gdata = json.load(f)
    gdata_org = deepcopy(gdata)
    gdata = {d["id"]: d for d in gdata}
    if "id" in data[0]:
        data = [d for d in data if d["id"] in gdata]
    print("$"*66)
    print(f"Questions file: {questions_file}")
    print(f"Data size: {len(data)}")

    if args.data_size and args.data_size < len(data):
        data = random.sample(data, args.data_size)
    for d in data:
        if args.if_process:
            if "response" in d and "output" not in d:
                response = d["response"]
            elif "output" in d:
                response = d["output"]
            if isinstance(response, list):
                response = "\n".join(response)
            d["pred"] = process_output_to_answer(response, args.reasoning_type, args.language)
            d["output"] = response
        d.pop("response", None)
            


    for di, d in enumerate(data):
        # d["pred"] = process_output_to_answer(d["response"])
        if "predict" in d and "pred" not in d:
            d["pred"] = d["predict"]
        if "answer" in d and "pred" not in d:
            d["pred"] = d["answer"].split("\n\n")
        if "PoT" in args.questions_file and "error" in ". ".join(d["pred"]).lower():
            d["pred"] = ["error"]
        if "id" not in d:
            d["id"] = gdata_org[di]["id"]
        if "question" not in d:
            d["question"] = gdata_org[di]["question"]

    evaluator = TaTQAEmAndF1()
    for d in data:
        if "pred_calculation" in d:
            gold = {
            "id": gdata[d["id"]]["id"],
            "question_type": gdata[d["id"]]["question_type"],
            "question": gdata[d["id"]]['question'],
            "paragraph": gdata[d["id"]]['paragraph'],
            "tables": gdata[d["id"]]['tables'],
            "reasoning": gdata[d["id"]]["reasoning"],
            "output": d.get("output", "").split("\n") if isinstance(d.get("output", ""), str) else d.get("output", [""]),
            "pred": d['pred'],
            "pred_calculation": d["pred_calculation"],
            "answer": gdata[d["id"]]['answer'] if args.language == "en" else gdata[d["id"]]["answer_c"],
            "answer_en": gdata[d["id"]]["answer"],
            "answer_type": gdata[d["id"]]['answer_type'],
            "scale": "",
        }
        else:
            gold = {
                "id": gdata[d["id"]]["id"],
                "question_type": gdata[d["id"]]["question_type"],
                "question": gdata[d["id"]]['question'],
                "paragraph": gdata[d["id"]]['paragraph'],
                "tables": gdata[d["id"]]['tables'],
                "reasoning": gdata[d["id"]]["reasoning"],
                "output": d.get("output", "").split("\n") if isinstance(d.get("output", ""), str) else d.get("output", [""]),
                "pred": d['pred'],
                "answer": gdata[d["id"]]['answer'] if args.language == "en" else gdata[d["id"]]["answer_c"],
                "answer_en": gdata[d["id"]]["answer"],
                "answer_type": gdata[d["id"]]['answer_type'],
                "scale": "",
            }
        evaluator(args.language, gold, d['pred'], "")
    print(str(evaluator))
    print(evaluator.get_overall_metric())

    
    data = evaluator.get_raw()

    em = [0, 0]
    f1 = [0, 0]
    for d in data:
        if all([len(a.split(" ")) <= 5 for a in d["answer_en"]]):
            em[0] += d["em"]
            em[1] += 1
        else:
            f1[0] += d["f1"]
            f1[1] += 1
    print(f"EM: {em[0]} / {em[1]}, {em[0] / em[1]}")
    print(f"F1: {f1[0]} / {f1[1]}, {f1[0] / f1[1]}")

    print(len(data))
    bertscore = [0, 0]
    P, R, F1 = score([". ".join(d["pred"]) for d in data], [". ".join(d["answer"]) for d in data], num_layers=12, model_type="bert-base-uncased", lang="en")
    for di, d in enumerate(data):
        d["bertscore"] = {"P": P[di].item(), "R": R[di].item(), "F1": F1[di].item()}
        if not all([len(a.split(" ")) <= 5 for a in d["answer_en"]]):
            bertscore[0] += d["bertscore"]["F1"]
            bertscore[1] += 1
    print(f"BertScore: {bertscore[0]} / {bertscore[1]}, {bertscore[0] / bertscore[1]}")
    # print(f"BertScore: P: {P.mean().item()}, R: {R.mean().item()}, F1: {F1.mean().item()}")

    if args.dump_file:
        with open(args.dump_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)




