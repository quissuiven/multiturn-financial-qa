import json
import argparse
import re
import csv
import sys
import os
from collections import defaultdict

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.program_utils import str_to_num, program_tokenization, equal_program
from src import config

def evaluate_predictions(gold_path, predictions_path, error_file_path="error_analysis.csv"):
    """Evaluates prediction file against the gold standard dataset."""
    with open(gold_path, 'r', encoding='utf-8') as f:
        gold_data = json.load(f)
    with open(predictions_path, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)

    if isinstance(gold_data, dict):
        flat_gold_data = [item for split in gold_data.values() for item in split]
    else:
        flat_gold_data = gold_data
    gold_dict = {item['id']: item for item in flat_gold_data}
    
    turn_exe_correct, turn_prog_correct, total_turns = 0, 0, 0
    sample_exe_correct, sample_prog_correct = 0, 0
    errors = defaultdict(list)

    for pred_item in pred_data:
        if 'error' in pred_item: continue
        gold_item = gold_dict.get(pred_item['id'])
        if not gold_item: continue

        gold_dialogue = gold_item.get('dialogue', {})
        pred_programs = pred_item.get('turn_program', [])
        pred_exe_ans = pred_item.get('executed_answers', [])
        
        num_turns = len(gold_dialogue.get('turn_program', []))
        if num_turns == 0:
            continue

        last_turn_exe_correct = False
        last_turn_prog_correct = False

        for i, gold_prog_str in enumerate(gold_dialogue.get('turn_program', [])):
            if i >= len(pred_programs) or i >= len(pred_exe_ans): continue
            
            total_turns += 1
            gold_exe_ans = gold_dialogue['executed_answers'][i]
            
            pred_ans = pred_exe_ans[i]
            is_exe_correct = False
            if isinstance(gold_exe_ans, str) and gold_exe_ans.lower() in ['yes', 'no']:
                if str(pred_ans).lower() == gold_exe_ans.lower():
                    is_exe_correct = True
            else:
                gold_ans_norm = round(str_to_num(gold_exe_ans), 5) if isinstance(str_to_num(gold_exe_ans), float) else str_to_num(gold_exe_ans)
                pred_ans_norm = round(pred_ans, 5) if isinstance(pred_ans, float) else pred_ans
                if pred_ans_norm == gold_ans_norm:
                    is_exe_correct = True
            
            if is_exe_correct:
                turn_exe_correct += 1

            is_program_correct = False
            gold_prog_tokenized = program_tokenization(gold_prog_str)
            pred_prog_tokenized = program_tokenization(pred_programs[i])
            if equal_program(gold_prog_tokenized, pred_prog_tokenized):
                is_program_correct = True
            else:
                def normalize_program_string(prog_str):
                    s = prog_str.replace(" ", "")
                    s = re.sub(r'(\d+)\.0(?![\d])', r'\1', s)
                    s = s.replace("const_", "")
                    return s
                gold_prog_norm = normalize_program_string(gold_prog_str)
                pred_prog_norm = normalize_program_string(pred_programs[i])
                if gold_prog_norm == pred_prog_norm:
                    is_program_correct = True
            
            if is_program_correct:
                turn_prog_correct += 1

            if not is_exe_correct or not is_program_correct:
                error_detail = {
                    "id": pred_item['id'],
                    "turn": i + 1,
                    "gold_program": gold_prog_str,
                    "predicted_program": pred_programs[i],
                    "gold_answer": gold_exe_ans,
                    "predicted_answer": pred_ans
                }
                if not is_exe_correct and not is_program_correct:
                    errors['both_mismatch'].append(error_detail)
                elif not is_exe_correct:
                    errors['answer_mismatch_only'].append(error_detail)
                else:
                    errors['program_mismatch_only'].append(error_detail)
            
            if i == num_turns - 1:
                last_turn_exe_correct = is_exe_correct
                last_turn_prog_correct = is_program_correct
        
        if last_turn_exe_correct:
            sample_exe_correct += 1
        if last_turn_prog_correct:
            sample_prog_correct += 1

    total_samples = len(pred_data)
    turn_exe_acc = (turn_exe_correct / total_turns) * 100 if total_turns > 0 else 0
    turn_prog_acc = (turn_prog_correct / total_turns) * 100 if total_turns > 0 else 0
    sample_exe_acc = (sample_exe_correct / total_samples) * 100 if total_samples > 0 else 0
    sample_prog_acc = (sample_prog_correct / total_samples) * 100 if total_samples > 0 else 0

    print("--- Evaluation Results ---")
    print(f"\n--- Sample-Level Accuracy ---")
    print(f"Total conversations evaluated: {total_samples}")
    print(f"  - Execution Accuracy: {sample_exe_acc:.2f}%")
    print(f"  - Program Accuracy:   {sample_prog_acc:.2f}%")

    print("\n--- Turn-Level Accuracy ---")
    print(f"Total conversational turns evaluated: {total_turns}")
    print(f"  - Execution Accuracy: {turn_exe_acc:.2f}%")
    print(f"  - Program Accuracy:   {turn_prog_acc:.2f}%")

    all_errors_flat = []
    for category, err_list in errors.items():
        for err in err_list:
            err['error_category'] = category
            all_errors_flat.append(err)
            
    with open(error_file_path, 'w', newline='', encoding='utf-8') as f:
        if all_errors_flat:
            writer = csv.DictWriter(f, fieldnames=all_errors_flat[0].keys())
            writer.writeheader()
            writer.writerows(all_errors_flat)
    print(f"\nFull error analysis saved to {error_file_path}")

    print("\n--- Turn-Level Error Analysis (Top 10) ---")
    for category, err_list in errors.items():
        print(f"\n--- {category.replace('_', ' ').title()} ({len(err_list)} errors) ---")
        if not err_list:
            print("  None")
        else:
            for err in err_list[:10]:
                print(f"  ID: {err['id']}, Turn: {err['turn']}")
                print(f"    Gold Prog: {err['gold_program']} (Exec: {err['gold_answer']})")
                print(f"    Pred Prog: {err['predicted_program']} (Exec: {err['predicted_answer']})")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions against a gold standard.")
    parser.add_argument("--gold_path", type=str, default=config.TEST_SET_PATH, help="Path to the gold standard JSON file.")
    parser.add_argument("--predictions_path", type=str, required=True, help="Path to the predictions JSON file.")
    parser.add_argument("--error_file_path", type=str, default=config.ANALYSIS_DIR / "error_analysis.csv", help="Path to save the error analysis CSV file.")
    args = parser.parse_args()

    evaluate_predictions(args.gold_path, args.predictions_path, args.error_file_path)

if __name__ == "__main__":
    main()
