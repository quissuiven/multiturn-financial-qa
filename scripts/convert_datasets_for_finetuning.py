import json
import os
import argparse
import sys

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.program_utils import dict_to_markdown_table
from src import config


def convert_to_openai_format(source_path, output_path):
    """
    Converts the sampled ConvFinQA data into the JSONL format required
    for OpenAI fine-tuning, preserving the order for traceability.
    """
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at {source_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {source_path}")
        return

    # Prepare to write to a .jsonl file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for sample in samples:
            # 1. Construct the System message
            doc = sample.get('doc', {})
            table_str = dict_to_markdown_table(doc.get('table', {}))
            system_content = (
                f"{doc.get('pre_text', '')}\n\n"
                f"TABLE:\n{table_str}\n\n"
                f"{doc.get('post_text', '')}"
            )
            
            messages = [{"role": "system", "content": system_content}]
            
            # 2. Construct the User/Assistant messages
            dialogue = sample.get('dialogue', {})
            questions = dialogue.get('conv_questions', [])
            programs = dialogue.get('turn_program', []) 

            for i, question in enumerate(questions):
                if i < len(programs):
                    messages.append({"role": "user", "content": question})
                    messages.append({"role": "assistant", "content": programs[i]})
            
            # 3. Write the final JSON object to a new line
            final_json_object = {"messages": messages}
            f_out.write(json.dumps(final_json_object) + '\n')

    print(f"Successfully converted {len(samples)} samples from {source_path} to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert dataset to OpenAI JSONL format for fine-tuning.")
    parser.add_argument("--train_source", type=str, default=config.TRAIN_SET_PATH, help="Path to the source training JSON file.")
    parser.add_argument("--train_output", type=str, default=config.TRAIN_SET_JSONL_PATH, help="Path to save the output training JSONL file.")
    parser.add_argument("--test_source", type=str, default=config.TEST_SET_PATH, help="Path to the source test JSON file.")
    parser.add_argument("--test_output", type=str, default=config.TEST_SET_JSONL_PATH, help="Path to save the output test JSONL file.")
    args = parser.parse_args()

    print("--- Preparing datasets in jsonl format for OpenAI Finetuning ---")
    convert_to_openai_format(args.train_source, args.train_output)
    convert_to_openai_format(args.test_source, args.test_output)
    print("\nData preparation complete.")
