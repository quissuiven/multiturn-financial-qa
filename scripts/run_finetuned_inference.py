import json
import os
import argparse
import datetime
import uuid
import sys
from tqdm import tqdm

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from src.program_utils import eval_program, program_tokenization, dict_to_2d_list_table
from src import config

def run_inference_and_process(model_id, source_json_path, output_path, limit: int = None):
    """
    Runs inference on a fine-tuned model, executes the predicted programs,
    and saves the results in an evaluation-ready format.
    Logs all traces to a single, unique run in LangSmith using a shared run_id.
    """
    # --- 1. Setup ---
    llm = ChatOpenAI(model=model_id, temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS)

    try:
        with open(source_json_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Source file not found at {source_json_path}")
        return

    if limit:
        print(f"Limiting processing to the first {limit} samples.")
        source_data = source_data[:limit]

    # --- 2. Inference and Processing Loop ---
    all_final_predictions = []
    print(f"Running inference and processing for {len(source_data)} samples with model: {model_id}")

    for sample in tqdm(source_data, desc="Processing samples"):
        sample_id = sample.get("id")
        doc = sample.get('doc', {})
        dialogue = sample.get('dialogue', {})
        
        table_str = dict_to_2d_list_table(doc.get('table', {}))
        system_content = f"{doc.get('pre_text', '')}\n\nTABLE:\n{table_str}\n\n{doc.get('post_text', '')}"
        
        questions = dialogue.get('conv_questions', [])
        predicted_programs = []
        executed_answers = []
        
        current_messages = [SystemMessage(content=system_content)]

        for i, question in enumerate(questions):
            current_messages.append(HumanMessage(content=question))
            
            try:
                response = llm.invoke(
                    current_messages, 
                    config={
                        "metadata": {"sample_id": sample_id, "turn": i+1},
                    }
                )
                program_str = response.content.strip()
                predicted_programs.append(program_str)
                
                current_messages.append(AIMessage(content=program_str))

                tokenized_prog = program_tokenization(program_str)
                _, exe_res = eval_program(tokenized_prog)
                executed_answers.append(exe_res)

            except Exception as e:
                print(f"\nError during API call or execution for sample {sample_id}, turn {i+1}: {e}")
                predicted_programs.append(f"[ERROR: {e}]")
                executed_answers.append("n/a")
                current_messages.append(AIMessage(content=f"[ERROR: {e}]"))

        all_final_predictions.append({
            "id": sample_id,
            "turn_program": predicted_programs,
            "executed_answers": executed_answers
        })

    # --- 3. Save Final Results ---
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_final_predictions, f, indent=4)
        print(f"\nInference and processing complete. Final predictions saved to {output_path}")
    except IOError as e:
        print(f"Error writing predictions to file: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference and processing on a fine-tuned OpenAI model with LangSmith tracing.")
    parser.add_argument("--model_id", type=str, default=config.FINETUNED_OPENAI_MODEL, help="The ID of the fine-tuned model.")
    parser.add_argument("--source_json_path", type=str, default=config.TEST_SET_PATH, help="Path to the source .json file.")
    parser.add_argument("--output_path", type=str, default=config.PREDICTIONS_DIR / "finetuned_on_test.json", help="Path to save the final, evaluation-ready predictions.")
    parser.add_argument("--limit", type=int, help="Limit the number of samples to process.")

    args = parser.parse_args()
    
    run_inference_and_process(
        args.model_id, 
        args.source_json_path, 
        args.output_path,
        args.limit
    )
