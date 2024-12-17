import os
import re
import sys
import json
import pandas as pd

sys.path.append('.')
a_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(a_directory)

from tqdm import tqdm
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Union

from utils.table import trans_table

def remove_invalid_chars(text: Union[str, List[List[str]]]) -> Union[str, List[List[str]]]:
    def remove_invalid_chars_str(text: str) -> str:
        # 正则表达式：保留ASCII字符（\x00-\x7F）或中文字符（\u4e00-\u9fff）
        valid_chars = re.compile(r'[\x00-\x7F\u4e00-\u9fff]+')
        
        # 找出并连接所有匹配的有效字符
        return ''.join(valid_chars.findall(text))
    # 正则表达式：保留ASCII字符（\x00-\x7F）或中文字符（\u4e00-\u9fff）
    if isinstance(text, str):
        return remove_invalid_chars_str(text)
    elif isinstance(text, list):
        return [[remove_invalid_chars_str(t) for t in row] for row in text]
    
    # 找出并连接所有匹配的有效字符
    return ''.join(valid_chars.findall(text))

def contains_chinese(text: Union[str, List[str]]):
    # 使用正则表达式匹配中文字符
    def contains_chinese_str(text: str):
        pattern = re.compile('[\u4e00-\u9fff]')
        return bool(pattern.search(text))
    if isinstance(text, str):
        return contains_chinese_str(text)
    elif isinstance(text, list):
        return any([contains_chinese_str(t) for t in text])


def unify_indentation(input_string):
    lines = input_string.splitlines()
    unified_lines = []
    
    for line in lines:
        # 检查行开头的空格数
        leading_spaces = len(line) - len(line.lstrip(' '))
        
        # 如果是2个或3个空格，替换为4个空格
        if leading_spaces == 2 or leading_spaces == 3:
            line = ' ' * 4 + line.lstrip(' ')
        
        unified_lines.append(line)

    return '\n'.join(unified_lines)

def process_output_to_answer(text, language):
    text = deepcopy(text)
    if language == "en":
        if "\nAnswer: \n" in text:
            text = text.split("\nAnswer: \n")[1].strip()
        if "\nAnswer:\n" in text:
            text = text.split("\nAnswer:\n")[1].strip()
        if "\nAnswer: " in text:
            text = text.split("\nAnswer: ")[1].strip()
        if "\nAnswer:" in text:
            text = text.split("\nAnswer:")[1].strip()
        if "Answer: " in text:
            text = text.split("Answer: ")[1].strip()
        if "Answer:" in text:
            text = text.split("Answer:")[1].strip()
    elif language == "zh":
        if "\n答案： \n" in text:
            text = text.split("\n答案： \n")[1].strip()
        if "\n答案：\n" in text:
            text = text.split("\n答案：\n")[1].strip()
        if "\n答案：" in text:
            text = text.split("\n答案：")[1].strip()
        if "\n答案:" in text:
            text = text.split("\n答案:")[1].strip()
        if "答案：" in text:
            text = text.split("答案：")[1].strip()
        if "答案:" in text:
            text = text.split("答案:")[1].strip()       
    text = text.replace("```json\n", " ").replace("```csv\n", " ").replace("```\n", " ").replace("```", " ").strip()
    return text

def formalize_answer(strr: str, language) -> str:
    if language == "en":
        answer = remove_invalid_chars(process_output_to_answer(unify_indentation(strr).replace("\r\n", "\n").replace("\\n\\n", "\n\n").strip(), language)).strip().strip("\n").strip()
    else:
        answer = process_output_to_answer(unify_indentation(strr).replace("\r\n", "\n").replace("\\n\\n", "\n\n").strip(), language).strip().strip("\n").strip()
    # if answer.split(" ") < 5:
    return answer


def is_number(s):
    try:
        # 尝试将字符串转换为 float
        float(s)
        return True
    except ValueError:
        return False
    

def get_answer_type(answer):
    if isinstance(answer, list):
        if all([is_number(a) for a in answer]):
            return "arithmetic"
        elif len(answer) > 1:
            return "multi-span"
        else:
            return "span"
    elif is_number(answer):
        return "arithmetic"
    else:
        return "span"
    
def process_output_to_reasoning(text, language):
    text = deepcopy(text)
    if language == "en":
        text = text.split("\nAnswer: \n")[0].split("\nAnswer:\n")[0].split("\nAnswer: ")[0].split("\nAnswer:")[0].strip()
        text = text.replace("Reasoning: \n", "").replace("Reasoning:\n", "").replace("Reasoning: ", "").replace("Reasoning:", "").strip()
        text = text.replace("Reason: \n", "").replace("Reason:\n", "").replace("Reason: ", "").replace("Reason:", "").strip()
    elif language == "zh":
        text = text.split("\n答案： \n")[0].split("\n答案：\n")[0].split("\n答案：")[0].split("\n答案:")[0].strip()
        text = text.replace("推理： \n", "").replace("推理：\n", "").replace("推理：", "").replace("推理:", "").strip()
    return text
    

    
def formalize_reasoning(strr: str, language) -> str:
    strr =  unify_indentation(strr).replace("\r\n", "\n").strip()
    strr.replace("\n\n", "\n").replace("**", " ")
    if language == "en":
        return remove_invalid_chars(process_output_to_reasoning(strr.strip(), language)).strip().strip("\n").strip()
    else:
        return process_output_to_reasoning(strr.strip(), language).strip().strip("\n").strip()


def split_markdown_tables(md_string):
    # 定义表格的分隔符模式，这里使用正则表达式来匹配表格的第二行，该行以 "|:---" 开头
    table_separator_pattern = r'\n\|:?-+.*\|:?-+.*\n'
    
    # 使用正则表达式找到分隔符的位置
    matches = list(re.finditer(table_separator_pattern, md_string))
    
    result_tables = []
    for i, match in enumerate(matches):
        # 获取表格的开始和结束索引
        start = match.end()  # 表格内容的开始为分隔符的结束
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(md_string)
        
        # 构造表格内容，包含分隔符前的一行（列名）和分隔符后的内容
        header_start = md_string.rfind('\n', 0, match.start()) + 1
        header = md_string[header_start:match.end()].strip()
        table_content = md_string[start:end].strip()
        
        # 泛化处理，判断表格内容末尾是否包含下一个表格的 header，header 以 | 开头并包含至少一个 |
        lines = table_content.splitlines()
        if i + 1 < len(matches) and len(lines) > 1 and re.match(r'^\|.*\|$', lines[-1]):
            table_content = '\n'.join(lines[:-1]).strip()
        
        # 将列名行与表格内容拼接
        result_tables.append(header + '\n' + table_content)

    # 返回分开的表格
    return result_tables


def unpack_response(response: str) -> Tuple[List[List[List[str]]], str]:
    # 移除冒号
    def remove_colon(s: str) -> str:
        if '：' in s:
            return s[s.index('：') + 1:].strip()
        if ':' in s:
            return s[s.index(':') + 1:].strip()
        return s

    response = response.strip()
    response = '\n'.join([line.strip() for line in response.split('\n')[1:-1]])
    response_part = response.split('\n\n---\n\n')
    response_tables = split_markdown_tables(response_part[0])
    response_text = response_part[1:]
    if isinstance(response_text, str):
        response_text = [response_text]
    # response_question = response_part[-3]
    # response_explanation = response_part[-2]
    # response_answer = response_part[-1]
    tables = []
    for response_table in response_tables:
        response_table_lines = response_table.split('|\n|')
        response_table_lines = [line.replace("\n", " ") for line in response_table_lines]
        response_table_lines = '|\n|'.join(response_table_lines).split('\n')
        response_table_lines = [response_table_lines[0]] + response_table_lines[2:]
        table = [line[1:-1].split('|') for line in response_table_lines]
        table = [[x.strip() for x in row] for row in table]
        tables.append(table)

    text = '\n\n---\n\n'.join([x.strip() for x in response_text])

    # question = remove_colon(response_question)
    # explanation = remove_colon(response_explanation)
    # answer = remove_colon(response_answer)

    return tables, text


def unpack_data(data, language: str) -> List[dict]:
    results = []
    for d in data:
        # if d['idx'] == '2533b8545f309e03':
        #     print(d['prompt'])
        #     print(d['response'])
        #     exit(0)

        try:
            table, text, question, explanation, answer = unpack_response(
                d['response'])
        except Exception as e:
            print(d['idx'], ':', e)
            print(d['response'])
            continue
        results.append({
            "source": {
                "qid": d['idx']
            },
            "table": {
                "content": table
            },
            "text": {
                "paragraph": text
            },
            "question": question,
            "explanation": explanation,
            "answer": answer,
            "translation": d['response'].split('\n')
        })
    return results


def form_paper(paragraphs: List[Dict[str, Any]], table_format: str) -> str:
    had_tables = []
    paper = ""
    for paragraph in paragraphs:
        paper += paragraph['text'] + '\n\n'
        if 'tables' in paragraph.keys() and paragraph['tables']:
            tables_list = [t for t in paragraph["tables"] if t["id"] not in had_tables]
            for t in tables_list:
                if "table" not in t.keys() and "extract_table" in t.keys():
                    t["table"] = t.pop("extract_table")
            tables = "\n\n".join([t["label"] + "\n" + t["caption"] + "\n" + trans_table(t["table"], table_format) for t in tables_list])
            had_tables += [t["id"] for t in tables_list]
            paper += tables + '\n\n'
    return paper



if __name__ == '__main__':
    result = []
    with open("./generate_answer/result/Llama3-70B/turn1/result_128.json", 'r', encoding='latin-1') as f:
        data = json.load(f)
    data = {d["id"]:d for d in data}
    with open("./generate_answer/result/GPT-4o/turn1/answer_modified_128.json", 'r', encoding='latin-1') as f:
        qa_data = json.load(f)
    # with open("./rule/results/text/main.w.table.json", 'r', encoding='latin-1') as f:
    #     data = json.load(f)
    for di, d in enumerate(qa_data):
        if d["id"] not in data.keys() and di > 63:
            d["annotated"] = False
            continue
        elif d["id"] not in data.keys():
            d["annotated"] = True
            d["if_remain"] = False
            continue
        d1 = data[d["id"]]    
        # d1["question"] = d1.pop("revised_question") if "revised_question" in d1.keys() else d1["question"]
        # d1["answer"] = d1.pop("revised_answer") if "revised_answer" in d1.keys() else d1["answer"]
        d1["answer"] = unify_indentation(d1["answer"].replace("\r\n", "\n"))
        # d1["reasoning"] = d1.pop("revised_reasoning") if "revised_reasoning" in d1.keys() else d1["predicted_answer"]
        d1["question_type"] = d1.pop("correct_question_type") if "correct_question_type" in d1.keys() else d1["question_type"]
        d1.pop("prediction", None)
        d1.pop("pred_answer", None)
        d1.pop("predicted_answer", None)
        d1.pop("prompt", None)
        d1.pop("score", None)
        d1["annotated"] = True
        d1["revised_question"] = d1["question"]
        d1["revised_answer"] = d1["answer"]
        d1["revised_reasoning"] = d1["reasoning"]
        qa_data[di] = d1
        
    
    with open('./generate_answer/result/GPT-4o/turn1/answer_modified_128.json', 'w', encoding='utf-8') as f:
        json.dump(qa_data, f, indent=4, ensure_ascii=False)
    # write_data_to_excel(result, './generate_query/result/turn1/result_gpt.xlsx', column_order)
