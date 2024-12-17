import os
import re
import sys
import ast
import json
import random
import argparse

sys.path.append('.')
a_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(a_directory)

from tqdm import tqdm
from copy import deepcopy
from transformers import set_seed
from typing import List, Dict, Any, Union, Tuple
random.seed(42)


PROMPT_ZERO_POT = """Table:
[Table]
Paragraph:
[PARAGRAPH]
Read the above Table and Paragraph, and then write code to answer the question "[QUESTION]".
Please **directly use** the information such as numbers in tables and paragraphs, do not define tables and then process them.
You must return the answer `ans = ` at the end of the code instead of `print`.
You cannot return just one or a few numbers or words, you must return a complete sentence.
""".strip()

def process_output_to_answer(tex: str, typee: str) -> List[str]:
    text = deepcopy(tex)
    scale = ""
    # print(f"Type: {typee}")
    if "CoT" in typee.lower() or "long" in typee:
        if "\nAnswer: \n" in text:
            text = text.split("\nAnswer: \n")[1].strip()
        if "\nAnswer:\n" in text:
            text = text.split("\nAnswer:\n")[1].strip()
        if "\nAnswer: " in text:
            text = text.split("\nAnswer: ")[1].strip()
        if "\nAnswer:" in text:
            text = text.split("\nAnswer:")[1].strip()
        text = text.replace("```json\n", " ").replace(
            "```csv\n", " ").replace("```\n", " ").replace("```", " ").strip()
        answer = text.split("\n\n")
    elif typee == "Pot" or "short" in typee or typee == "pot":
        # print("Short")
        text = text.strip().strip("\n")
        answer = parse_answer(fix_answer(extract_code(text)))
        if isinstance(answer, str):
            answer = [answer]
    elif typee == "direct":
        text = text.strip().strip("\n")
        answer = text.split("\n\n")
    return answer


if __name__ == '__main__':
    from utils.table import trans_table
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
        PROMPT = PROMPT_ZERO_POT
        if args.retrieve == "short_context":
            if not d["tables"]:
                PROMPT = PROMPT.replace("Table:\n[Table]\n", "").replace("Table and ", "")
            if not d["paragraph"]:
                PROMPT = PROMPT.replace("Paragraph:\n[PARAGRAPH]\n", "").replace(" and Paragraph", "")
        table_format = "list"
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
        # d["response"] = "Here is the completed function:\n```python\n"
        d["response"] = "```python\n"
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
            record.append((p[0], process_output_to_answer(p[0], "short"), p[1]))
        d["output"], d["pred"] = consistency(record)
        d.pop("response", None)
        d["output"] = d["output"].split("\n")
    print(data[-1]["output"])

    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

