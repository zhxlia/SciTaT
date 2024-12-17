import re
import json
import pandas as pd

from copy import deepcopy
from typing import List, Dict, Any


class Table(object):
    def __init__(self, table: List[List[str]], table_name: str, table_type: str):
        def fix_table(table: List[List[str]]) -> List[List[str]]:
            def replace_sql_keywords(text: str) -> str:
                # SQL 保留字及其近义词
                sql_keywords = {
                    "select": "choose",
                    "from": "source",
                    "where": "filter",
                    "when": "during",
                    "and": "plus",
                    "or": "either",
                    "group": "collect",
                    "to": "toward"
                }
                # 正则表达式模式，用于匹配整个单词
                pattern = r'\b({})\b'.format('|'.join(sql_keywords.keys()))
                # 替换函数

                def replace(match) -> str:
                    word = match.group(0).lower()  # 转换为小写
                    return sql_keywords.get(word, word)  # 获取近义词，如果不存在则返回原词
                # 使用正则表达式替换文本中的保留字
                return re.sub(pattern, replace, text, flags=re.IGNORECASE)

            table = deepcopy(table)
            for i, k in enumerate(table[0]):
                # fix column names
                if self.table_type == 'sql':
                    k = replace_sql_keywords(k.lower())
                    k = k.replace("#", "num")
                    k = re.sub(r'[^a-zA-Z0-9]', '_', k)
                    k = re.sub(r'_+', '_', k)
                    k = k.strip().strip('_')
                    if bool(re.match(r'^\d', k)):
                        k = 'column_' + k
                table[0][i] = k
            # Rename empty column names to "column" followed by its index
            for i, k in enumerate(table[0]):
                if not k:  # Checks if the column name is empty
                    table[0][i] = f'column{i}'
            for row in table[1:]:
                row = [str(x).replace('\n', ' ') if x else x for x in row]
            return table

        self.table = table
        self.table_name = table_name
        self.table_type = table_type
        if isinstance(self.table[0], list):
            self.table = fix_table(self.table)

    def to_markdown(self) -> str:
        headers = self.table[0]
        header_str = " | ".join(header for header in headers)
        rows = self.table[1:]
        row_strs = [(" | ".join(str(item) for item in row)) for row in rows]
        return "\n".join([header_str] + row_strs)

    def to_database(self) -> str:
        def all_elements_are_numbers(lst: List[str]) -> bool:
            def is_number(s):
                if not s:
                    return False
                try:
                    float(s)
                    return True
                except ValueError:
                    return False
            return all(is_number(item) for item in lst)

        column_names: List[str] = [
            k.replace(' ', '_').lower() for k in self.table[0]]
        all_columns: List[str] = []
        for i, k in enumerate(column_names):
            v = [row[i] for row in self.table[1:] if i < len(row)]
            all_number_flag = all_elements_are_numbers(v)
            if not all_number_flag:
                v = [f'"{item}"' for item in v]
            all_columns.append(
                f"{k} {'int' if all_number_flag else 'text'}")
        all_values: List[str] = []
        for i, k in enumerate(column_names):
            v = [str(row[i]) for row in self.table[1:] if i < len(row)]
            all_values.append(f"{k}: {', '.join(v)} ;")
        return f"CREATE TABLE {self.table_name} (\n" + ' ,\n'.join(all_columns) + '\n);\n' + '/*\n' + 'Columns and examples in each column :\n' + '\n'.join(all_values) + '\n*/'

    def to_dict(self) -> str:
        return json.dumps(self.table, ensure_ascii=False, indent=4)

    def __str__(self) -> str:
        if self.table_type == 'nl':
            return self.to_markdown()
        elif self.table_type == 'pl':
            return self.to_dict()
        elif self.table_type == 'sql':
            return self.to_database()
        else:
            raise Exception(f"illegal table type: {self.table_type}")


def list_to_dict(table) -> List[Dict[str, str]]:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    # fix empty header
    for idx, header in enumerate(table[0]):
        header = header.strip()
        if not header:
            table[0][idx] = f"<unnamed>"
    # pad table row into same length
    max_len = max(len(row) for row in table)
    for row in table:
        row.extend(['-' for _ in range(max_len - len(row))])
    # fix duplicate header
    for idx, header in enumerate(table[0]):
        if table[0].count(header) > 1:
            table[0][idx] = f"{header} ({idx})"
    return [dict(zip(table[0], row)) for row in table[1:]]

def list_to_dict_str(table, truncated: bool = False) -> str:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    # fix empty header
    for idx, header in enumerate(table[0]):
        header = header.strip()
        if not header:
            table[0][idx] = f"<unnamed>"
    # pad table row into same length
    max_len = max(len(row) for row in table)
    for row in table:
        row.extend(['-' for _ in range(max_len - len(row))])
    # fix duplicate header
    for idx, header in enumerate(table[0]):
        if table[0].count(header) > 1:
            table[0][idx] = f"{header} ({idx})"
    result = [dict(zip(table[0], row)) for row in table[1:]]
    if truncated and len(result) > 10:
        truncated_result = result[:3] + ["..."] + result[-3:]
        result = json.dumps(truncated_result, ensure_ascii=False, indent=4)
        return result
    else:
        return json.dumps(result, ensure_ascii=False, indent=4)
        


def list_to_plain_md(table: List[Dict[str, Any]]) -> str:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    result = ''
    # if "caption" in table[0].keys():
    #     result += "## " + ", ".join(table[0]["caption"].split(" | ")) + "\n"
    result += "\n".join([" | ".join([str(c) for c in row]) for row in table])
    return result


def list_to_md(table: List[Dict[str, Any]], truncated: bool = False) -> str:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    if len(table) == 0:
        return ""
    result = ''
    if isinstance(table[0], dict) and "caption" in table[0].keys():
        result += "## " + ", ".join(table[0]["caption"].split(" | ")) + "\n"
    result += "| " + " | ".join([str(c) for c in table[0]]) + " |\n"
    if len(table) > 1:
        result += "|" + ":---|" * len(table[0]) + "\n| "
    if truncated and len(table[0]["table"]) > 15:
        result += " |\n| ".join([" | ".join([str(c) for c in row]) for row in table[1:5]]) + " |"
        result += "\n...\n| "
        result += " |\n| ".join([" | ".join([str(c) for c in row]) for row in table[-4:]]) + " |"
    else:
        result += " |\n| ".join([" | ".join([str(c) for c in row]) for row in table[1:]]) + " |"
    return result


def list_to_html(table: List[Dict[str, List[List[str]]]]) -> str:
    """
    Generate an HTML string for a table with a title.
    
    :param table_data: A list of lists where each inner list represents a row in the table.
    :param table_title: The title to be displayed above the table.
    :return: A string containing HTML for the table.
    """
    if isinstance(table[0], dict):
        table = table[0]["table"]
    table_title = table[0]["caption"] if "caption" in table[0] else ""
    indent = "    "  # Define the indentation (4 spaces in this case)
    # indent = "\t"  # Define the indentation (4 spaces in this case)
    
    # Start the HTML string with a table title (if provided)
    html_str = f"<h2>{table_title}</h2>\n" if table_title else ""
    
    # Begin the table
    html_str += "<table>\n"
    
    # Add table header
    if table:
        html_str += f"{indent}<thead>\n{indent}{indent}<tr>\n"
        for header in table[0]:  # Assuming the first row contains headers
            html_str += f"{indent}{indent}{indent}<th>{header}</th>\n"
        html_str += f"{indent}{indent}</tr>\n{indent}</thead>\n"
        
    # Add table body
    html_str += f"{indent}<tbody>\n"
    for row in table[1:]:  # Skip the first row since it's assumed to be the header
        html_str += f"{indent}{indent}<tr>\n"
        for cell in row:
            html_str += f"{indent}{indent}{indent}<td>{cell}</td>\n"
        html_str += f"{indent}{indent}</tr>\n"
    html_str += f"{indent}</tbody>\n"
    
    # End the table
    html_str += "</table>"
    
    return html_str


def list_to_html_no_space(table: List[Dict[str, List[List[str]]]]) -> str:
    """
    Generate an HTML string for a table with a title.
    
    :param table_data: A list of lists where each inner list represents a row in the table.
    :param table_title: The title to be displayed above the table.
    :return: A string containing HTML for the table.
    """
    if isinstance(table[0], dict):
        table = table[0]["table"]
    table_data = table[0]["table"]
    table_title = table[0]["caption"] if "caption" in table[0] else ""
    
    # Start the HTML string with a table title (if provided)
    html_str = f"<h2>{table_title}</h2>\n" if table_title else ""
    
    # Begin the table
    html_str += "<table>\n"
    
    # Add table header
    if table_data:
        html_str += "    <thead>\n        <tr>"
        for header in table_data[0]:  # Assuming the first row contains headers
            html_str += f"<th>{header}</th>"
        html_str += "</tr>\n    </thead>\n"
        
    # Add table body
    html_str += "    <tbody>\n"
    for row in table_data[1:]:  # Skip the first row since it's assumed to be the header
        html_str += "        <tr>"
        for cell in row:
            html_str += f"<td>{cell}</td>"
        html_str += "</tr>\n"
    html_str += "    </tbody>\n"
    
    # End the table
    html_str += "</table>"
    
    return html_str


def list_to_text(table: List[Dict[str, List[List[str]]]]) -> str:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    text = ""
    for row in table[1:]:
        text += f"The {table[0][0]} {row[0]} has the "
        text += ", the ".join([f"{table[0][i]} {row[i]}" for i in range(1, len(table[0]))])
        text += ". "
    return text


def list_to_tuple(table: List[Dict[str, List[List[str]]]]) -> str:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    result = ""
    # if "caption" in table[0].keys():
    #     result += f"# Table Title\nTitle: {table[0]['caption']}\n"
    result += f"# Column Header\nColumn header: "
    result += " ".join([f"(T, 0, {i}, {i+1}, {table[0][i]})" for i, ci in enumerate(table[0])])
    result += "\n# Row Header\nRow header: "
    result += " ".join([f"(L, 0, {i}, {i+1}, {table[i][0]})" for i, ci in enumerate(table)])
    result += "\n# Non-Header\nNon-header: "
    result += "\n".join([" ".join([f"(C, {i}, {j}, {table[i][j]})" for j in range(1, len(table[i]))]) for i in range(1, len(table))])
    return result

def list_to_csv(table: List[Dict[str, List[List[str]]]]) -> str:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    result = ""
    # if "caption" in table[0].keys():
    #     result += f"## {table[0]['caption']}\n"
    result += "\n".join([", ".join([c for c in row]) for row in table])
    return result


def list_to_tsv(table: List[Dict[str, List[List[str]]]], truncated: bool = False) -> str:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    result = ""
    # if "caption" in table[0].keys():
    #     result += f"## {table[0]['caption']}\n"
    if truncated and len(table) > 11:
        result += "\n".join(["    ".join([c for c in row]) for row in table[:4]])
        result += "\n...\n"
        result += "\n".join(["    ".join([c for c in row]) for row in table[-3:]])
    else:
        result += "\n".join(["    ".join([c for c in row]) for row in table])
    return result


def list_to_pd_org(table: List[Dict[str, List[List[str]]]]) -> str:
    table = table[0]["table"]
    result = "pd.DataFrame([\n    [\n"
    result += "\n    ],\n    [\n".join([",\n".join([f"        \"{c}\"" for c in row]) for row in table])
    result += "\n], columns = [\n"
    result += ",\n".join([f"        \"{h}\"" for h in table[0]])
    result += "\n]\n)"
    return result

def list_to_pd(table: List[Dict[str, List[List[str]]]]) -> str:
    if isinstance(table[0], dict):
        table = table[0]["table"]
    result = f"pd.DataFrame({json.dumps(table[1:], ensure_ascii=False, indent=4)}, columns = {json.dumps(table[0], ensure_ascii=False, indent=4)}\n)"
    # result = pd.DataFrame(table[1:], columns = table[0])
    return result


def trans_table(org_table: List[Any], typee: str, truncated: bool = False):
    if isinstance(org_table[0], dict):
        org_table = org_table[0]["table"]
    if "plain_md" in typee:
        tr_table = list_to_plain_md(org_table)
    elif "md" in typee:
        tr_table = list_to_md(org_table)
    elif "htmlns" in typee:
        tr_table = list_to_html_no_space(org_table)
    elif "html" in typee:
        tr_table = list_to_html(org_table)
    elif "text" in typee:
        tr_table = list_to_text(org_table)
    elif "tuple" in typee:
        tr_table = list_to_tuple(org_table)
    elif "db" in typee:
        tr_table = str(Table(org_table, 'information', "sql"))
    elif "csv" in typee:
        tr_table = list_to_csv(org_table)
    elif "tsv" in typee:
        tr_table = list_to_tsv(org_table)
    elif "pd" in typee: 
        tr_table = list_to_pd(org_table)
    elif "dict" in typee:
        tr_table = json.dumps(list_to_dict(org_table), ensure_ascii=False, indent=4)
        # tr_table = list_to_dict(org_table)
    else:
        # tr_table = org_table
        tr_table = json.dumps(org_table, ensure_ascii=False, indent=4)
    return tr_table


def process_utterance_and_table(utterance: str, table: List[Dict[str, List[List[str]]]], typee: str = "plm") -> (str, str):
    # 1. 找到同时出现在utterance中的词也在table中出现的，并分别在utterance和table中在这些词后面打标记“ [M]”
    # 忽视utterance后面可能有的符号，比如, . ?，并把[M]标记在符号前
    def mark_word(word, text):
        # 使用正则表达式替换单词，但忽略单词后面的标点符号
        pattern = re.compile(r'(\b' + re.escape(word) + r'\b)(?=[,.?]*\s|$)')
        return pattern.sub(r'\1 [M]', text)
    if isinstance(table[0], dict):
        table = table[0]["table"]
    # 生成table中的所有单词集合
    words_in_table = {word for row in table for word in row}
    # 分割utterance中的单词
    utterance_words = utterance.split()
    updated_utterance = utterance

    # 更新utterance中的单词
    for word in utterance_words:
        clean_word = re.sub(r'[,.?]', '', word)
        if clean_word in words_in_table:
            updated_utterance = mark_word(clean_word, updated_utterance)
    
    # 更新table中的单词
    updated_table = []
    for row in table:
        # print(row)
        updated_row = []
        for word in row:
            if word in utterance_words:
                updated_row.append(word + " [M]")
            else:
                updated_row.append(word)
        updated_table.append(updated_row)
    
    # 将table转换为指定格式的字符串
    if typee == "plm":
        table_str = "<T> "
        for row in updated_table:
            table_str += "<R> " + " | ".join(row) + " | "
    else:
        table_str = "<T> " + trans_table(updated_table, typee)
    
    # 将utterance前面加上<Q>
    updated_utterance = "<Q> " + updated_utterance
    
    # return updated_utterance, table_str.strip()
    return f"{updated_utterance}\n{table_str.strip()}"




# def trans_table(org_table: List[Dict[str, Any]], args):
#     # if "pd" in type and "demo" not in args.prompt_type: 
#     #     org_table = list_to_pd(org_table)
#     if "dict" in type:
#         org_table = list_to_dict(org_table)
#     elif "list" in type:
#         org_table = org_table[0]["table"]
#     return org_table


if __name__ == '__main__':
    with open("./dataset/TabFact/test.list.json", "r", encoding="utf8") as f:
        data: List[Dict[str, Any]] = json.load(f)

    for d in data:
        # d["table"] = str(Table(d["table"][0]["table"], 'information', "sql"))
        if d["utterance"] == "natalia strelkova come in last place with 112.28 point":
            d["table"] = list_to_dict_str(d["table"])
            # d["table"] = str(Table(d["table"][0]["table"], 'information', "sql"))
            print(d["table"])
            break
        # d["source"] = process_utterance_and_table(d["utterance"], d["table"], "plm").split("\n")
        # print(d["source"])
        # break
        # if "[M]" in d["source"]:
        #     print(d["source"])
        #     break

    # with open(f"./dataset/TabFact/test.marked.json", 'w', encoding='utf-8') as json_file:
    #     json.dump(data, json_file, ensure_ascii=False, indent=4)
    

