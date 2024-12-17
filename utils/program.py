import re

from typing import Dict, List
from func_timeout import func_timeout, FunctionTimedOut


def replace_function_name_and_add_result_assignment_0(code_str, new_name):
    """
    从输入的代码字符串中提取函数名并替换成指定的新名称，
    并在最后一个函数调用前增加 `function_result = new_name(...)` 赋值语句，保留调用时的参数。

    参数:
    code_str (str): 包含函数定义及调用的代码字符串
    new_name (str): 新的函数名

    返回:
    str: 替换和添加结果赋值后的代码字符串
    """
    # 使用正则表达式提取函数定义中的函数名
    function_name_match = re.search(r'def\s+(\w+)\s*\(', code_str)
    if not function_name_match:
        code_str = "def solve():\n" + "\n".join([f"    {c}" for c in code_str.split("\n")]) + "\n    return ans"
        function_name_match = re.search(r'def\s+(\w+)\s*\(', code_str)
        # raise ValueError("未找到函数定义，请检查输入的代码字符串。")
    
    # 获取原函数名
    original_name = function_name_match.group(1)
    
    # 替换所有出现的函数名
    replaced_code = re.sub(rf'\b{original_name}\b', new_name, code_str)
    
    # 找到最后一次函数调用的位置
    call_matches = list(re.finditer(rf'\b{new_name}\s*\((.*?)\)', replaced_code, re.DOTALL))
    if call_matches:
        last_call = call_matches[-1]
        start, end = last_call.span()  # 最后一次调用的起始和结束位置
        arguments = last_call.group(1)  # 提取最后一次调用的参数内容
        
        # 在最后一次调用前插入 "function_result = new_name(arguments)"
        result_assignment = f"function_result = {new_name}({arguments})"
        replaced_code = replaced_code[:start] + result_assignment + replaced_code[end:]
    
    return replaced_code


def replace_function_name_and_add_result_assignment(code_str, new_name):
    """
    从输入的代码字符串中提取函数名并替换成指定的新名称，
    并在最后一个函数调用前增加 `function_result = new_name(...)` 赋值语句，保留调用时的参数。
    参数:
    code_str (str): 包含函数定义及调用的代码字符串
    new_name (str): 新的函数名
    返回:
    str: 替换和添加结果赋值后的代码字符串
    """
    # 使用正则表达式提取函数定义中的函数名
    function_name_match = re.search(r'def\s+(\w+)\s*\(', code_str)
    if not function_name_match:
        code_str = "def solve():\n" + "\n".join([f"    {c}" for c in code_str.split("\n")]) + "\n    return ans\nfunction_result = solve()"
        function_name_match = re.search(r'def\s+(\w+)\s*\(', code_str)
    
    # 获取原函数名
    original_name = function_name_match.group(1)
    
    # 替换所有出现的函数名
    replaced_code = re.sub(rf'\b{original_name}\b', new_name, code_str)
    
    # 找到所有的函数调用位置，但排除函数定义的情况
    call_matches = list(re.finditer(rf'\b{new_name}\s*\((.*?)\)', replaced_code, re.DOTALL))
    
    # 过滤掉函数定义中的调用，确保我们不会在函数定义中插入赋值语句
    function_definitions = list(re.finditer(rf'def\s+{new_name}\s*\(', replaced_code))
    
    # 获取所有函数定义的结束位置
    definition_end_positions = [match.end() for match in function_definitions]
    
    # 找到最后一次有效的函数调用位置
    last_valid_call = None
    for call in reversed(call_matches):
        call_start, call_end = call.span()
        # 如果函数调用出现在函数定义之后，跳过
        if any(call_start > end_pos for end_pos in definition_end_positions):
            last_valid_call = call
            break

    if last_valid_call:
        start, end = last_valid_call.span()  # 最后一次有效调用的起始和结束位置
        arguments = last_valid_call.group(1)  # 提取最后一次调用的参数内容
        
        # 在最后一次有效调用前插入 "function_result = new_name(arguments)"
        result_assignment = f"function_result = {new_name}({arguments})"
        replaced_code = replaced_code[:start] + result_assignment + replaced_code[end:]
    
    return replaced_code



def replace_newline_in_print(code_str):
    """
    替换代码字符串中 print 语句里的 '\n' 为一个空格
    """
    # 定义正则模式，匹配 print 语句内的 '\n'
    pattern = r"(print\s*\(\".*?)(\n)(.*?\"\))"
    
    # 使用正则替换 '\n' 为 ' '
    modified_code = re.sub(pattern, lambda m: m.group(1) + " " + m.group(3), code_str)
    
    return modified_code


def extract_code(code: str) -> str:
    if code.count("```") % 2 == 1:
        code  = "Here is the completed function:\n```python\n" + code
    # print(code)
    # print("c"*55)
    code = code.split("# Example utterance")[-1]
    
    pattern = re.compile(r'```python(.*?)```', re.DOTALL)
    matches = pattern.findall(code)
    
    # Return the last match if there are any
    if matches:
        code = matches[-1].strip()
    # code = replace_newline_in_print(code)
    code = code.split('\n')
    code = [c for c in code if c and not c.startswith("#")]
    return '\n'.join(code)


def fix_answer(function: str) -> str:
    function = replace_newline_in_print(function)
    function = replace_function_name_and_add_result_assignment(function, "solve")
    # append_blank = True
    function_sentences = function.split('\n')
    # # print(function_sentences)
    # # if function.startswith("def solver"):
    function_start = [f for f in function_sentences if f and not f.startswith("import")]
    function_import = [f for f in function_sentences if f.startswith("import")]
    if function_start[0].startswith("def"):
        function_sentences = [function_start[0]] + [f"    {f}" for f in function_import] + function_start[1:]
    #     append_blank = False
    # if "=" in function_sentences[-1]:
    #     function_sentences[-1] = f"ans = {function_sentences[-1].split('=')[-1].strip()}"
    # for f in function_sentences:
    #     print(f)
    # print("%"*99)
    # if append_blank:
    #     function = '\n'.join([f"    {fu}" for fu in function_sentences])
    # else:
    #     function = '\n'.join(function_sentences)
    function = '\n'.join(function_sentences)
    program = f"""{function}
""".strip()
    # print(program)
    # print("%"*99)
    # print(append_blank)
    return program


def parse_answer(program: str, time_out: float = 5) -> str:
    def run_exec(program: str) -> str:
        try:
            local_scope = {}
            exec(program, {}, local_scope)
            return str(local_scope.get("function_result", None))
        except Exception as e:
            return f"Error occurred: {e}"

    try:
        return func_timeout(time_out, run_exec, args=(program,))
    except FunctionTimedOut:
        return "Error: Execution time exceeded the limit"
    except Exception as e:
        return f"Error occurred: {e}"
    
if __name__=="__main__":
    sent = '''
def offensive_tweet_count_and_performance():
    ratios = [61, 17.5, 14, 10.6, 10.2, 10, 10, 9, 9]
    counts = [61, 35, 14, 106, 92, 10, 10, 278, 9]
    combined_count = sum(counts)
    overall_performance = \"A-Sub3 had the highest F1-Macro score of our task A systems\"
    ans = f\"Combined count of offensive tweets for all terms that have a ratio of 10 or higher: {combined_count}\\n\" \\
          f\"Contribution to overall task performance: {overall_performance}\"
    return ans
ans = offensive_tweet_count_and_performance()
'''
# print("Anomalies in WMT14 EN->DE task:")
# print(wmt14_anomalies)
# print("\nAnomalies in IWSLT14 DE->EN task:")
# print(iws14_anomalies)
# '''

#     print(replace_newline_in_print(try_code))
    es = extract_code(sent)
    # print(es)

    fa = fix_answer(es)
    print(fa)

    pa = parse_answer(fa)
    print(pa)
