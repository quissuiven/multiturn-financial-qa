import json
import argparse
import os 
import re
import sys
from typing import List, Dict, Optional

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.program_utils import eval_program, program_tokenization, dict_to_2d_list_table
from src import config

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = config.OPENAI_MODEL
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL_NAME = config.GEMINI_MODEL

def list_2d_to_markdown_table(table_data: List[List[str]]) -> str:
    """
    Converts a 2D table (list of lists) into markdown table format.
    
    Input: List of lists where first sublist is headers, rest are data rows
    Example: [["Name", "Age"], ["Alice", "25"], ["Bob", "30"]]
    
    Output: Markdown-formatted table string
    Example: "Name | Age\n--- | ---\nAlice | 25\nBob | 30"
    """
    if not table_data: return "No table provided."
    try:
        header = " | ".join(map(str, table_data[0]))
        separator = " | ".join("---" for _ in table_data[0])
        rows = [" | ".join(map(str, row)) for row in table_data[1:]]
        return "\n".join([header, separator] + rows)
    except Exception:
        return "Error formatting table."

# --- Prompt Construction ---
def construct_program_generation_prompt(pre_text: str, table_str: str, post_text: str, history: str, question: str) -> str:
    return (
        "You are a reasoning agent. Your task is to generate a single program string to answer the user's question based on the provided context and conversation history.\n\n"
        "**CRITICAL RULES:**\n"
        "1.  **Analyze Intent**: First, determine if the current question is a follow-up that uses the *result* of the previous turn, or if it is a new, independent question.\n"
        "2.  **Program Construction**:\n"
        "    - **If the question builds on the previous result** (e.g., 'what is the percentage change?'), you MUST copy the program from the previous turn (available in the history) and append the new operation.\n"
        "    - **If the question is independent** (e.g., 'what about in 2008?'), you MUST start a new program from scratch.\n"
        "3.  **Program Type**: Decide if the answer is a direct number from the text or requires a calculation.\n"
        "    - If direct, the program is just the number (e.g., `306870`).\n"
        "    - If calculation, you MUST use one of these 6 operations: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`.\n"
        "4.  **Show Your Work**: Do NOT pre-calculate values. If the answer requires subtracting 50 from 100, the program must be `subtract(100, 50)`, not `50`.\n"
        "5.  **Sequential Steps ONLY**: Do NOT nest operations. Programs must be a sequence of single operations separated by commas.\n"
        "6.  **Use Step References**: For multi-step calculations, you MUST use the `#n` syntax to refer to the result of a previous step.\n"
        "7.  **Subtraction Order**: The `subtract(a, b)` operation computes `a - b`. For 'the change from 2007 to 2008', if 2007 is 100 and 2008 is 120, the program is `subtract(120, 100)`.\n\n"
        "**CORRECT, SEQUENTIAL FORMAT EXAMPLE:**\n"
        "To calculate `(100 - 50) / 50`, the program MUST be: `subtract(100, 50), divide(#0, 50)`\n\n"
        "**INCORRECT, NESTED FORMAT EXAMPLE:**\n"
        "`divide(subtract(100, 50), 50)` <-- DO NOT DO THIS.\n\n"
        "**INCORRECT, PRE-CALCULATED EXAMPLE:**\n"
        "`divide(50, 50)` <-- DO NOT DO THIS.\n\n"
        "--- FEW-SHOT EXAMPLES ---\n\n"
        "**Example 1:**\n"
        "Conversation History:\n"
        "Turn 1:\nQ: what is the net change in rent expense from 2003 to 2004?\nA: 4785000.0\nProgram: subtract(118741000, 113956000)\n\n"
        "Current Question: what percentage change does this represent?\n"
        "Correct Program: subtract(118741000, 113956000), divide(#0, 113956000)\n\n"
        "**Example 2:**\n"
        "Conversation History:\n"
        "Turn 1:\nQ: what was the total number of shares purchased in 11/07?\nA: 2891719.0\nProgram: 2891719\n\nTurn 2:\nQ: and the average price paid per share for that time?\nA: 44.16\nProgram: 44.16\n\nTurn 3:\nQ: so what was the total amount paid for these shares?\nA: 127698311.04\nProgram: multiply(2891719, 44.16)\n\n"
        "Current Question: and converted to the hundreds?\n"
        "Correct Program: multiply(2891719, 44.16), divide(#0, const_1000000)\n\n"
        "--- END OF EXAMPLES ---\n\n"
        "**YOUR TASK:**\n"
        "Construct a single program string for the current turn. Your output MUST be ONLY the program string and nothing else.\n\n"
        f"== Pre-Table Context ==\n{pre_text}\n\n"
        f"== Table Data ==\n{table_str}\n\n"
        f"== Post-Table Context ==\n{post_text}\n\n"
        f"== Conversation History (Question, Answer, and Program) ==\n{history if history else 'No history yet.'}\n\n"
        f"== Current Question ==\n{question}\n\n"
        "Program:"
    )

# --- LLM Interaction ---
def call_llm(llm_choice: str, prompt: str) -> str:
    """Calls the specified LLM using LangChain and returns the text response."""
    try:
        if llm_choice == "openai":
            if not OPENAI_API_KEY: return "[ERROR: OPENAI_API_KEY not set]"
            model = ChatOpenAI(model=OPENAI_MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0)
        elif llm_choice == "gemini":
            if not GEMINI_API_KEY: return "[ERROR: GEMINI_API_KEY not set]"
            model = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, google_api_key=GEMINI_API_KEY, temperature=0)
        else:
            return "[ERROR: LLM not implemented]"

        chain = ChatPromptTemplate.from_template("{prompt}") | model | StrOutputParser()
        return chain.invoke({"prompt": prompt})
    except Exception as e:
        return f"[ERROR: LangChain LLM call failed - {e}]"

# --- Main Processing Logic ---
def run_baseline_inference(llm_choice: str, input_path: str, output_path: str, limit: Optional[int] = None):
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    all_final_outputs = []
    data_items = dataset if isinstance(dataset, list) else next(iter(dataset.values()), [])

    if limit:
        print(f"Limiting processing to the first {limit} samples.")
        data_items = data_items[:limit]

    for item in data_items:
        item_id = item.get('id')
        print(f"\nProcessing ID: {item_id}")

        doc = item.get('doc', {})
        table_data = dict_to_2d_list_table(doc.get('table', {}))
        table_str = list_2d_to_markdown_table(table_data)
        
        questions = item.get('dialogue', {}).get('conv_questions', [])
        
        history = ""
        turn_programs, executed_answers = [], []

        for i, question in enumerate(questions):
            print(f"  Turn {i+1}: Generating program...")
            
            prog_prompt = construct_program_generation_prompt(doc.get('pre_text', ''), table_str, doc.get('post_text', ''), history, question)
            program_str = call_llm(llm_choice, prog_prompt).strip()
            turn_programs.append(program_str)
            
            tokenized_prog = program_tokenization(program_str)
            _, exe_res = eval_program(tokenized_prog)
            executed_answers.append(exe_res)
            
            history += f"Turn {i+1}:\nQ: {question}\nA: {exe_res}\nProgram: {program_str}\n\n"

        all_final_outputs.append({
            "id": item_id,
            "turn_program": turn_programs,
            "executed_answers": executed_answers
        })
        print(f"  Finished processing for {item_id}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_final_outputs, f, indent=4)
    print(f"\nBatch generation complete. Predictions saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate baseline predictions using a few-shot prompted model.")
    parser.add_argument("--llm", type=str, required=True, choices=["openai", "gemini"])
    parser.add_argument("--input_data_path", type=str, default=config.TEST_SET_PATH, help="Path to the input data JSON file.")
    parser.add_argument("--output_path", type=str, default=config.PREDICTIONS_DIR / "baseline_on_test.json", help="Path to save the output predictions.")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process.")
    args = parser.parse_args()

    run_baseline_inference(args.llm, args.input_data_path, args.output_path, args.limit)

if __name__ == "__main__":
    main()
