import re
from sympy import simplify

all_ops = ["add", "subtract", "multiply", "divide", "exp", "greater"]

def str_to_num(text):
    text = str(text).replace(",", "").strip()
    try:
        return float(text)
    except ValueError:
        if "%" in text:
            text = text.replace("%", "")
            try:
                return float(text) / 100.0
            except ValueError:
                return "n/a"
        if "const_" in text:
            text = text.replace("const_", "")
            if text == "m1": text = "-1"
            try:
                return float(text)
            except ValueError:
                return "n/a"
        return "n/a"

def process_row(row_in):
    row_out = []
    for num_str in row_in:
        num = str(num_str).replace("$", "").split("(")[0].strip()
        num = str_to_num(num)
        if num == "n/a":
            return "n/a"
        row_out.append(num)
    return row_out

def eval_program(program):
    """
    Calculates the numerical result of a program.
    """
    this_res = "n/a"
    try:
        if not program or program[-1] != 'EOF':
            return 1, "n/a"
        program = program[:-1]

        if len(program) == 1:
            num = str_to_num(program[0])
            if num != "n/a":
                return 0, num
        
        for ind, token in enumerate(program):
            if ind % 4 == 0 and token.strip("(") not in all_ops: return 1, "n/a"
            if (ind + 1) % 4 == 0 and token != ")": return 1, "n/a"

        program_str = "|".join(program)
        steps = program_str.split(")")[:-1]
        res_dict = {}

        for ind, step in enumerate(steps):
            step = step.strip()
            op = step.split("(")[0].strip("|").strip()
            args = step.split("(")[1].strip("|").strip().split("|")
            arg1_str, arg2_str = args[0].strip(), args[1].strip()

            if op in ["add", "subtract", "multiply", "divide", "exp", "greater"]:
                arg1_str, arg2_str = args[0].strip(), args[1].strip()
                arg1 = res_dict[int(arg1_str[1:])] if "#" in arg1_str else str_to_num(arg1_str)
                arg2 = res_dict[int(arg2_str[1:])] if "#" in arg2_str else str_to_num(arg2_str)
                if arg1 == "n/a" or arg2 == "n/a": return 1, "n/a"
                
                op_map = {
                    "add": lambda a, b: a + b, "subtract": lambda a, b: a - b,
                    "multiply": lambda a, b: a * b, "divide": lambda a, b: a / b if b != 0 else "n/a",
                    "exp": lambda a, b: a ** b, "greater": lambda a, b: "yes" if a > b else "no"
                }
                this_res = op_map[op](arg1, arg2)
                if this_res == "n/a": return 1, "n/a"

            res_dict[ind] = this_res
        
        if isinstance(this_res, float):
            this_res = round(this_res, 5)
        return 0, this_res
    except Exception:
        return 1, "n/a"

def program_tokenization(original_program):
    original_program = original_program.split(', ')
    program = []
    for tok in original_program:
        cur_tok = ''
        for c in tok:
            if c == ')':
                if cur_tok != '':
                    program.append(cur_tok)
                cur_tok = ''
            cur_tok += c
            if c in ['(', ')']:
                program.append(cur_tok)
                cur_tok = ''
        if cur_tok != '':
            program.append(cur_tok)
    program.append('EOF')
    return program

def dict_to_2d_list_table(table_dict: dict) -> list[list[str]]:
    """
    Converts a nested dictionary to a 2D table format (list of lists).
    
    Input: Nested dict where outer keys become column headers, inner keys become row headers
    Example: {"Q1": {"Revenue": 100, "Profit": 20}, "Q2": {"Revenue": 120, "Profit": 25}}
    
    Output: List of lists representing a table with transposed layout
    Example: [["", "Revenue", "Profit"], ["Q1", "100", "20"], ["Q2", "120", "25"]]
    """
    if not table_dict or not isinstance(table_dict, dict):
        return []
    
    header = [""] + list(table_dict.keys())
    rows = {}
    for col_header, col_data in table_dict.items():
        for row_header, cell_value in col_data.items():
            if row_header not in rows:
                rows[row_header] = {}
            rows[row_header][col_header] = cell_value
            
    if not rows: return [header]

    row_headers = sorted(rows.keys())
    output_table = [header]
    for r_header in row_headers:
        row = [r_header] + [rows[r_header].get(c_header, "") for c_header in header[1:]]
        output_table.append(row)
        
    return output_table

def dict_to_markdown_table(table_data: dict) -> str:
    """
    Directly converts a nested dictionary to markdown table format.
    
    Input: Nested dict where outer keys become row headers, inner keys become column headers  
    Example: {"Q1": {"Revenue": 100, "Profit": 20}, "Q2": {"Revenue": 120, "Profit": 25}}
    
    Output: Markdown-formatted table string
    Example: " | Revenue | Profit\n--- | --- | ---\nQ1 | 100 | 20\nQ2 | 120 | 25"
    """
    if not table_data:
        return "No table provided."
    
    try:
        headers = [""] + list(next(iter(table_data.values())).keys())
        rows = []
        for col_header, row_data in table_data.items():
            row = [col_header] + [str(v) for v in row_data.values()]
            rows.append(row)

        header_str = " | ".join(headers)
        separator_str = " | ".join("---" for _ in headers)
        row_strs = [" | ".join(row) for row in rows]
        
        return "\n".join([header_str, separator_str] + row_strs)
    except Exception as e:
        print(f"Warning: Could not format table for a sample. Error: {e}")
        return str(table_data)

def equal_program(program1, program2):
    def symbol_recur(step, step_dict, sym_map):
        op, args_str = step.split("(", 1)
        op = op.strip("|").strip()
        args = args_str.strip(")").strip("|").strip().split("|")
        arg1, arg2 = args[0].strip(), args[1].strip()
        
        arg1_part = symbol_recur(step_dict[int(arg1[1:])], step_dict, sym_map) if "#" in arg1 else sym_map[arg1]
        arg2_part = symbol_recur(step_dict[int(arg2[1:])], step_dict, sym_map) if "#" in arg2 else sym_map[arg2]
        
        op_map = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "exp": "**", "greater": ">"}
        return f"( {arg1_part} {op_map[op]} {arg2_part} )"

    try:
        sym_map, sym_ind = {}, 0
        p1_steps = "|".join(program1[:-1]).split(")")[:-1]
        step_dict_1 = {i: s.strip() + ")" for i, s in enumerate(p1_steps)}

        for step in step_dict_1.values():
            op, args_str = step.split("(", 1)
            args = args_str.strip(")").strip("|").strip().split("|")
            for arg in args:
                if "#" not in arg and arg not in sym_map: sym_map[arg] = f"a{sym_ind}"; sym_ind += 1
        
        p2_steps = "|".join(program2[:-1]).split(")")[:-1]
        step_dict_2 = {i: s.strip() + ")" for i, s in enumerate(p2_steps)}

        sym_prog1 = simplify(symbol_recur(p1_steps[-1] + ")", step_dict_1, sym_map))
        sym_prog2 = simplify(symbol_recur(p2_steps[-1] + ")", step_dict_2, sym_map))
        
        return sym_prog1 == sym_prog2
    except Exception:
        return False
