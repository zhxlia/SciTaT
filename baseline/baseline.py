import os
import re
import sys
import ast
import json
import random
import argparse

sys.path.append('.')

from tqdm import tqdm
from copy import deepcopy
from bert_score import score
from transformers import set_seed
from typing import List, Dict, Any, Union, Tuple
random.seed(42)

from utils.program import fix_answer, parse_answer, extract_code


PROMPT_ZERO_DIRECT= '''Based on the information in the Table and Paragraph, you should answer the question.
If there are multiple questions, you need to answer them one by one, and the answers are separated by "\n\n".

Table (including its label, caption and content):
[Table]
Paragraph:
[PARAGRAPH]

Please answer the question "[QUESTION]".
'''.strip()


PROMPT_ZERO_COT = '''Based on the information in the Table and Paragraph, you should answer the question.
Represent your answer with: "Reason: <Your Reason>\nAnswer: <Your Answer>\n".
If there are multiple questions, you need to answer them one by one, and the answers are separated by "\n\n".

Table (including its label, caption and content):
[Table]
Paragraph:
[PARAGRAPH]

Please answer the question "[QUESTION]".
'''.strip()

DEMO_COT_FORMAT = '''Table:
[Table] (including its label, caption and content):
Paragraph:
[PARAGRAPH]
Based on the above demonstrations，please answer the question "[QUESTION]".
Reason: <Your Reason>
Answer: <Your Answer>
'''.strip()


PROMPT_FEW_COT = '''Based on the information in the Table and Paragraph, you should answer the question.
Represent your answer with: "Reason: <Your Reason>\nAnswer: <Your Answer>".
Whenever possible, use single spans rather than sentences as <Your Answer>.
If there are multiple questions, you need to answer them one by one, and the answers are separated by "\n\n".
Here are some demonstrations you may refer to.

---

[EXAMPLES]

---

Table:
[Table]
Paragraph:
[PARAGRAPH]
Based on the above demonstrations, please answer the question "[QUESTION]".
'''.strip()


PROMPT_ZERO_POT = """Table:
[Table]
Paragraph:
[PARAGRAPH]
Read the above Table and Paragraph, and then write code to answer the question "[QUESTION]".
Please **directly use** the information such as numbers in tables and paragraphs, do not define tables and then process them.
You must return the answer `ans = ` at the end of the code instead of `print`.
Attention that if there are multiple questions, you need to answer them one by one, and the answers are separated by "\n\n".
""".strip()


DEMO_POT_FORMAT = '''Read the following text and table, and then write code to answer a question and provide the unite.
[PARAGRAPH]
[Table]
Quesetion: [QUESTION]
<Your Answer>
'''


PROMPT_FEW_POT = """Read the following text and table, and then write code to answer the question.
Here are some demonstrations you may refer to.

---

[EXAMPLES]

---

Based on the above demonstrations, read the following text and table, and then write code to answer a question:
You must return the answer `ans = ` at the end of the code instead of `print`.
[PARAGRAPH]
[Table]
Quesetion: [QUESTION]
""".strip()

prompt_map = {"en": {"long": {0: [PROMPT_ZERO_COT], 1: [PROMPT_FEW_COT, DEMO_COT_FORMAT]}, "short": {0: [PROMPT_ZERO_POT], 1: [PROMPT_FEW_POT, DEMO_POT_FORMAT]}, "direct": {0: [PROMPT_ZERO_DIRECT]}}}



def extract_and_join(input_string):
    # 使用正则表达式提取<>中的内容
    matches = re.findall(r'<(.*?)>', input_string)
    
    # 如果有匹配的内容，则用逗号连接并返回
    if matches:
        return ','.join(matches)
    else:
        return input_string


def process_output_to_answer(tex: str, typee: str, language: str) -> List[str]:
    text = deepcopy(tex)
    # print(f"Type: {typee}")
    if "cot" in typee.lower() or "long" in typee:
        if language == "en":
            if "\nAnswer: \n" in text:
                text = text.split("\nAnswer: \n")[1].strip()
            elif "\nAnswer:\n" in text:
                text = text.split("\nAnswer:\n")[1].strip()
            elif "Answer: " in text:
                text = text.split("Answer: ")[1].strip()
            elif "Answer:" in text:
                text = text.split("Answer:")[1].strip()
            text = text.replace("```json\n", " ").replace(
                "```csv\n", " ").replace("```\n", " ").replace("```", " ").strip()
            answer = text.split("\n\n")
            answer = [extract_and_join(a) for a in answer]
        elif language == "zh":
            if "答案:" in text:
                text = text.split("答案:")[1].strip().strip("\n").strip()
            elif "答案：" in text:
                text = text.split("答案：")[1].strip().strip("\n").strip()
            if "答案是:" in text:
                text = text.split("答案是:")[1].strip().strip("\n").strip()
            elif "答案是：" in text:
                text = text.split("答案是：")[1].strip().strip("\n").strip()      
            elif "答案是" in text:
                text = text.split("答案是")[1].strip().strip("\n").strip()          
            elif "答案" in text:
                text = text.split("答案")[1].strip().strip("\n").strip()
            text = text.replace("```json\n", " ").replace(
                "```csv\n", " ").replace("```\n", " ").replace("```", " ").strip()
            answer = text.split("\n\n")
            answer = [extract_and_join(a) for a in answer]
    elif "pot" in typee.lower() or "short" in typee:
        # print("Short")
        text = text.strip().strip("\n")
        answer = parse_answer(fix_answer(extract_code(text)))
        if isinstance(answer, str):
            answer = [answer]
    elif typee == "direct" or typee == "sum":
        text = text.strip().strip("\n")
        answer = text.split("\n\n")
    else:
        print(f"Type: {typee}")
    return answer


if __name__ == '__main__':
    from utils.table import trans_table
    from utils.tatqa_metric import TaTQAEmAndF1
    from utils.program import fix_answer, parse_answer, extract_code
    from utils.formalize_annotated import remove_invalid_chars, form_paper
    from utils.retrieve import bm25_similarity_multiprocess_diff_docs, evaluate_recall
    from utils.generator import generate_with_llm, consistency

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="llm path")
    parser.add_argument("--config_file", type=str, help="config path")
    parser.add_argument("--questions_file", type=str, help="questions file")
    parser.add_argument("--language", type=str, default="en", help="language", choices=["en", "zh"])
    parser.add_argument("--retrieve", type=str, help="if retrieve", choices=["bm25", "long_context", "short_context"])
    parser.add_argument("--reasoning_type", type=str, help="reasoning type", choices=["CoT", "PoT", "direct"])
    parser.add_argument("--papers_file", type=str, help="papers file")
    parser.add_argument("--tables_file", type=str, help="tables file")
    parser.add_argument("--dump_file", type=str, help="dump path")
    parser.add_argument("--data_size", type=int, help="data size")
    parser.add_argument("--paragraph_top_k", type=int, help="paragraph top k")
    parser.add_argument("--table_top_k", type=int, help="table top k")
    parser.add_argument("--shot_num", type=int, default=1, help="shot num")
    parser.add_argument("--demo_obtained", type=str,
                        help="demo obtained", choices=["bm25", "fixed", "combination"])
    parser.add_argument("--demo_file", type=str, help="demo file")
    parser.add_argument("--random_seed", type=int,
                        default=42, help="random seed")
    args = parser.parse_args()
    set_seed(args.random_seed)

    model = args.model
    config_file = args.config_file
    questions_file = args.questions_file
    dump_file = args.dump_file
    data_size = args.data_size

    
    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if args.papers_file:
        with open(args.papers_file, 'r', encoding='utf-8', errors='ignore') as f:
            papers = json.load(f)
    if args.tables_file:
        with open(args.tables_file, 'r', encoding='utf-8', errors='ignore') as f:
            table_data = json.load(f)
        tables = {}
        for t in table_data:
            if t["paper_id"] not in tables.keys():
                tables[t["paper_id"]] = []
            tables[t["paper_id"]].append(t)
    if args.demo_file:
        with open(args.demo_file, 'r', encoding='utf-8') as f:
            demo_data = json.load(f)
        # retrieve the demos
        if args.shot_num > 0 and args.demo_obtained == "bm25":
            docs = [d["question"] + "\n".join([p["text"] for p in d["paragraphs"]]) + str(
                d["table"]["table"]) for d in demo_data]
            queries = [d["question"] + "\n".join(
                [p["text"] for p in d["paragraphs"]]) + str(d["table"]["table"]) for d in data]
            retrs = bm25_similarity_multiprocess_diff_docs(
                docs, queries, args.shot_num)
            all_demos = [[demo_data[d0[0]] for d0 in d] for di, d in enumerate(retrs)]
    
    print(f"Data size: {len(data)}")
    if args.data_size and args.data_size < len(data):
        data = random.sample(data, args.data_size)

    # Retrieve the paragraphs and tables from the whole paper
    if args.retrieve == "bm25" and args.papers_file and args.tables_file:
        print("Retrieving...")
        all_docs = [[p["text"] for p in papers[d["paper_id"]]] for d in data]
        # print(all_docs[0])
        print(len(all_docs))
        all_tables = [[t["label"] + "\n" + t["caption"] + "\n" + trans_table(t["extract_table"], "list") for t in tables[d["paper_id"]]] for d in data]
        # print(all_tables[0][0])
        print(len(all_tables))
        retrieve_result = bm25_similarity_multiprocess_diff_docs(all_docs, [d["question"] for d in data], args.paragraph_top_k)
        # print(retrieve_result[0])
        docs = [[all_docs[di][d0[0]] for d0 in d] for di, d in enumerate(retrieve_result)]
        retrieve_result_t = bm25_similarity_multiprocess_diff_docs(all_tables, [d["question"] for d in data], args.table_top_k)
        tables = [[all_tables[di][d0[0]] for d0 in d] for di, d in enumerate(retrieve_result_t)]
        gold_texts = []
        for d in data:
            if d["paragraph"]:
                gold_texts.append([d["paragraph"]["text"]])
            else:
                gold_texts.append([])
        paragraph_recall, single_paragraph_recalls = evaluate_recall([i for i in range(1, args.paragraph_top_k + 1) if i % 2 != 0], docs, gold_texts)
        table_recall, single_table_recalls = evaluate_recall([i for i in range(1, args.table_top_k + 1) if i % 2 != 0], tables, [[trans_table(t["table"], "list") for t in d["tables"]] for d in data])
        print(f"Paragraph recall: {paragraph_recall}")
        print(f"Table recall: {table_recall}")

    # Generate the prompts
    for di, d in tqdm(enumerate(data), desc="Processing data"):
        question = d["question"] if args.language == "en" else d["question_c"]
        d["response"] = ""
        if "paper_id" not in d:
            d["paper_id"] = "-".join(d["id"].split("-")[:-1])
        if args.reasoning_type == "CoT":
            d["pred_type"] = "long"
        elif args.reasoning_type == "PoT":
            d["pred_type"] = "short"
        elif args.reasoning_type == "direct":
            d["pred_type"] = "direct"
        PROMPT = prompt_map[args.language][d['pred_type']][1 if args.shot_num else 0][0]
        if args.retrieve == "short_context":
            if not d["tables"]:
                PROMPT = PROMPT.replace("Table:\n[Table]\n", "").replace("Table and ", "")
            if not d["paragraph"]:
                PROMPT = PROMPT.replace("Paragraph:\n[PARAGRAPH]\n", "").replace(" and Paragraph", "")
        table_format = "plain_md" if d["pred_type"] == "long" or d["pred_type"] == "direct" or d["pred_type"] == "ours_long" else "list"
        if args.retrieve == "bm25" and args.papers_file and args.tables_file:
            dprompt = PROMPT.replace("[Table]", "\n".join(tables[di])).replace("[PARAGRAPH]", "\n\n".join(docs[di]))
            d["recall"] = {"paragraph": single_paragraph_recalls[di], "table": single_table_recalls[di]}
        elif args.retrieve == "long_context" and args.papers_file:
            paper = "\n\n".join([p["text"] for p in papers[d["paper_id"]]])
            # dprompt = PROMPT.replace("[Table]", "\n".join([t["label"] + "\n" + t["caption"] + "\n" + trans_table(t["table"], table_format) for t in d["tables"]])).replace("[PARAGRAPH]", paper)
            dprompt = PROMPT.replace("Table (including its label, caption and content):\n[Table]\nParagraph:\n[PARAGRAPH]", form_paper(papers[d["paper_id"]], table_format))
        else:
            paragraph = d["text"] if "text" in d else d["paragraph"].get("text", "")
            dprompt = PROMPT.replace("[Table]", "\n".join([t["label"] + "\n" + t["caption"] + "\n" + trans_table(t["table"], table_format) for t in d["tables"]])).replace("[PARAGRAPH]", paragraph)
        dprompt = dprompt.replace("[QUESTION]", question)
        if "short" in d["pred_type"]:
            d["response"] = "Here is the completed function:\n```python\n"
        if args.shot_num > 0:
            DEMO_FORMAT = prompt_map[d['pred_type']][1 if args.shot_num else 0][1]
            if args.demo_obtained == "fixed" and len(demo_data[d["pred_type"]]) >= args.shot_num:
                demos = random.sample(demo_data[d["pred_type"]], args.shot_num)
            elif args.demo_obtained == "fixed" and len(demo_data[d["pred_type"]]) < args.shot_num:
                demos = demo_data[d["pred_type"]]
            elif args.demo_obtained == "bm25":
                demos = all_demos[di]
            if d["pred_type"] == "long" or d["pred_type"] == "ours_long" or d["pred_type"] == "direct":
                dprompt = dprompt.replace("[EXAMPLES]", "\n\n---\n\n".join([deepcopy(DEMO_FORMAT).replace("[Table]", '\n'.join([trans_table(t["table"], table_format) for t in demo["tables"]])).replace(
                    "[PARAGRAPH]", demo["paragraph"].get("text", "")).replace("[QUESTION]", demo["question"]).replace(
                    "<Your Answer>", "\n\n".join(demo["answer"])).replace("<Your Reason>", demo["reasoning"]) for demo in demos]))
            else:
                dprompt = dprompt.replace("[EXAMPLES]", "\n\n---\n\n".join([deepcopy(DEMO_FORMAT).replace("[Table]", '\n'.join([trans_table(t["table"], table_format) for t in demo["tables"]])).replace(
                    "[PARAGRAPH]", demo["paragraph"].get("text", "")).replace("[QUESTION]", demo["question"]).replace(
                    "<Your Answer>", demo["code"]) for demo in demos]))
        d["instruction"] = remove_invalid_chars(dprompt)

    print(data[-1]["instruction"])
    with open(dump_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    # '''
    with open(config_file, 'r', encoding='latin-1') as f:
        config = json.load(f)
    
    predictions = generate_with_llm(model, data, config, 'chat')

    for di, d in enumerate(data):
        record: List[Tuple[str, str, float]] = []
        for pi, p in enumerate(predictions[di]):
            record.append((p[0], process_output_to_answer(p[0], d["pred_type"], args.language), p[1]))
        d["output"], d["pred"] = consistency(record)
    print(data[-1]["output"])

    evaluator = TaTQAEmAndF1()
    for d in data:
        gold = {
            "id": d["id"],
            "question_type": d["question_type"],
            "question": d['question'],
            "paragraph": d['paragraph'],
            "tables": d['tables'],
            "reasoning": d["reasoning"],
            "pred": d['pred'],
            "output": d['output'].split("\n"),
            "answer": d['answer'] if args.language == "en" else d['answer_c'],
            "answer_en": d['answer'],
            "answer_type": d['answer_type'],
            "scale": "",
            "pred_type": d["pred_type"]
        }
        evaluator(args.language, gold, d['pred'], "")
    print(str(evaluator))
    print(evaluator.get_overall_metric())
    em = [0, 0]
    f1 = [0, 0]
    for d in evaluator.get_raw():
        if all([len(a.split(" ")) <= 5 for a in d["answer_en"]]):
            em[0] += d["em"]
            em[1] += 1
        else:
            f1[0] += d["f1"]
            f1[1] += 1
    print(f"EM: {em[0]} / {em[1]}, {em[0] / em[1]}")
    print(f"F1: {f1[0]} / {f1[1]}, {f1[0] / f1[1]}")
    bertscore = [0, 0]
    P, R, F1 = score([". ".join(d["pred"]) for d in data], [". ".join(d["answer"]) for d in evaluator.get_raw()], num_layers=12, model_type="bert-base-uncased", lang="en")
    for di, d in enumerate(evaluator.get_raw()):
        d["bertscore"] = {"P": P[di].item(), "R": R[di].item(), "F1": F1[di].item()}
        if not all([len(a.split(" ")) <= 5 for a in d["answer_en"]]):
            bertscore[0] += d["bertscore"]["F1"]
            bertscore[1] += 1
    print(f"BertScore: {bertscore[0]} / {bertscore[1]}, {bertscore[0] / bertscore[1]}")
    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(evaluator.get_raw(), f, ensure_ascii=False, indent=4)

