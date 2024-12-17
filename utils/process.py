import re
import regex
import hashlib

from typing import List, Dict, Any, Union

def remove_latex_comments(latex_code: Union[str, List[str]]) -> Union[str, List[str]]:
    # 定义一个函数，处理每一行，保留以 \% 转义的百分号
    def process_line(line: str) -> str:
        # 使用正则表达式查找未被转义的百分号
        pattern = r'(?<!\\)%'
        # 找到第一个未被转义的百分号
        match = re.search(pattern, line)
        if match:
            # 如果找到了百分号，保留该百分号之前的内容
            return line[:match.start()], True
        else:
            # 如果没有找到百分号，直接返回整行
            return line, False
        
    comment_block_pattern = re.compile(r'\\begin{comment}.*?\\end{comment}\n?', re.DOTALL)
    # 对每一行进行处理
    # processed_lines = [process_line(line) for line in latex_code.splitlines() if process_line(line).strip()]
    def process_str(code: str):
        code = code.strip()
        code = re.sub(comment_block_pattern, '', code)
        processed_lines = [process_line(line)[0] for line in code.splitlines() if (process_line(line)[0].strip() and process_line(line)[1]) or not process_line(line)[1]] 
        # 返回处理后的代码，并保持换行符
        return "\n".join(processed_lines)
    
    if not latex_code or (isinstance(latex_code, dict) and not latex_code.keys()):
        return latex_code
    
    if isinstance(latex_code, str):
        return process_str(latex_code)
    elif isinstance(latex_code, dict) and "text" in latex_code.keys():
        latex_code["text"] = process_str(latex_code["text"])
        return latex_code
    elif isinstance(latex_code, list) and isinstance(latex_code[0], dict) and "text" in latex_code[0].keys():
        for c in latex_code:
            c["text"] = process_str(c["text"])
        return latex_code
    else:
        latex_code = [line.strip() for line in latex_code]
        latex_code = [re.sub(comment_block_pattern, '', line) for line in latex_code]
        processed_lines = [process_line(line)[0] for line in latex_code if (process_line(line)[0].strip() and process_line(line)[1]) or not process_line(line)[1]] 
        # 返回处理后的代码，并保持换行符
        return processed_lines
    

def find_closing_brace_fixed(s, start_index):
        stack = 1  # 已经遇到了一个 '{'
        print(f"start: {s[start_index]}")
        for i in range(start_index, len(s)):
            print(s[i])
            if s[i] == '{':
                stack += 1
                print(f"stack add: {stack}")
            elif s[i] == '}':
                stack -= 1
                print(f"stack sub: {stack}")
                if stack == 0:
                    return i
        return -1  # 未找到匹配的右大括号

def extract_captions_labels(text, span: str):
    matches = []
    if span == "caption":
        caption_pattern = r"\\caption{"
    elif span == "label":
        caption_pattern = r"\\label{"
    for match in re.finditer(caption_pattern, text):
        start = match.end()  # 起始大括号后的索引
        end = find_closing_brace_fixed(text, start)
        if end != -1:
            # 提取完整的 caption 内容，包括嵌套大括号的内容
            matches.append(text[start:end])
    return matches


def generate_hash_code(input_string: str, algorithm: str = 'sha256') -> str:
    """
    生成指定字符串的哈希值。

    参数:
    input_string (str): 需要生成哈希的字符串
    algorithm (str): 哈希算法，默认使用 'sha256'，可选 'md5', 'sha1', 'sha256', 'sha512'

    返回:
    str: 生成的哈希值
    """
    # 选择哈希算法
    try:
        hash_function = getattr(hashlib, algorithm)
    except AttributeError:
        raise ValueError(f"不支持的哈希算法: {algorithm}")
    
    # 生成哈希值
    hash_object = hash_function()
    hash_object.update(input_string.encode('utf-8'))  # 将字符串转换为字节，并更新哈希对象
    return hash_object.hexdigest()


if __name__ == '__main__':
    text = "\caption{{\\bf Base Filters}.}"
    print(extract_captions_labels(text, "caption"))