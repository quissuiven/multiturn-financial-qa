import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def create_final_datasets(
    source_path,
    train_path,
    test_path,
    train_size=1000,
    test_size=200,
    random_state=42
):
    """
    Creates final training and test sets using a precise, multi-stage sampling strategy
    to ensure proportional representation of all specified rare categories.
    """
    try:
        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at {source_path}")
        return

    # --- 1. Flatten and Pre-process Data into a DataFrame ---
    all_samples = [item for split in data.values() for item in split]
    
    processed_data = []
    for item in all_samples:
        features = item.get('features', {})
        dialogue = item.get('dialogue', {})
        programs_str = str(dialogue.get('turn_program', []))
        processed_data.append({
            'id': item['id'],
            'has_type_2_question': features.get('has_type2_question', False),
            'num_dialogue_turns': features.get('num_dialogue_turns', 0),
            'has_exp': 'exp' in programs_str,
            'has_greater': 'greater' in programs_str,
            'full_data': item
        })
        
    df = pd.DataFrame(processed_data)
    val_ratio = test_size / (train_size + test_size)

    # --- Initialize final train/test dataframes and remaining pool ---
    final_train_dfs = []
    final_test_dfs = []
    remaining_df = df.copy()

    def split_and_assign(condition, category_name, manual_test_size=None, stratify_col=None):
        nonlocal remaining_df
        
        # Select from the current pool of remaining samples
        df_to_split = remaining_df[condition].copy()
        
        if df_to_split.empty:
            print(f"Stage: No samples found for '{category_name}'. Skipping.")
            return

        # Remove these samples from the main pool
        remaining_df = remaining_df.drop(df_to_split.index)

        if manual_test_size is not None:
            # Manual split for fixed numbers
            train_df, test_df = train_test_split(df_to_split, test_size=manual_test_size, random_state=random_state)
        else: # Proportional split
            stratify_on = df_to_split[stratify_col] if stratify_col and len(df_to_split[stratify_col].unique()) > 1 else None
            train_df, test_df = train_test_split(
                df_to_split, test_size=val_ratio, random_state=random_state, stratify=stratify_on
            )
        
        final_train_dfs.append(train_df)
        final_test_dfs.append(test_df)
        print(f"Stage: Split {len(df_to_split)} '{category_name}' samples -> {len(train_df)} train, {len(test_df)} test.")

    # --- 2. Execute the multi-stage splitting as defined ---
    split_and_assign(remaining_df['has_exp'], 'exp operation', manual_test_size=1)
    split_and_assign(remaining_df['num_dialogue_turns'] == 9, '9-turn dialogues', manual_test_size=1)
    split_and_assign(remaining_df['has_greater'], 'greater operation')
    split_and_assign(remaining_df['num_dialogue_turns'] == 8, '8-turn dialogues')
    split_and_assign(remaining_df['num_dialogue_turns'] == 7, '7-turn dialogues')
    split_and_assign(remaining_df['num_dialogue_turns'] == 1, '1-turn dialogues')

    # --- 3. Final Stage: Stratify the rest ---
    current_train_count = sum(len(d) for d in final_train_dfs)
    current_test_count = sum(len(d) for d in final_test_dfs)
    
    remaining_train_needed = train_size - current_train_count
    remaining_test_needed = test_size - current_test_count

    remaining_df['stratify_key'] = remaining_df.apply(
        lambda row: f"type2_{row['has_type_2_question']}_turns_{row['num_dialogue_turns']}",
        axis=1
    )
    
    value_counts = remaining_df['stratify_key'].value_counts()
    to_remove = value_counts[value_counts < 2].index
    remaining_df_filtered = remaining_df[~remaining_df['stratify_key'].isin(to_remove)]

    common_train_df, common_test_df = train_test_split(
        remaining_df_filtered,
        train_size=remaining_train_needed,
        test_size=remaining_test_needed,
        stratify=remaining_df_filtered['stratify_key'],
        random_state=random_state
    )
    final_train_dfs.append(common_train_df)
    final_test_dfs.append(common_test_df)
    print(f"Stage: Filled remaining slots with {len(common_train_df)} train and {len(common_test_df)} test samples.")

    # --- 4. Combine and Save Final Datasets ---
    final_train_df = pd.concat(final_train_dfs)
    final_test_df = pd.concat(final_test_dfs)

    train_json = final_train_df['full_data'].tolist()
    test_json = final_test_df['full_data'].tolist()

    os.makedirs(os.path.dirname(train_path), exist_ok=True)
    with open(train_path, 'w', encoding='utf-8') as f:
        json.dump(train_json, f, indent=4)
    print(f"\nFinal training set with {len(train_json)} samples saved to {train_path}")

    with open(test_path, 'w', encoding='utf-8') as f:
        json.dump(test_json, f, indent=4)
    print(f"Final test set with {len(test_json)} samples saved to {test_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create final training and test sets.")
    parser.add_argument("--source_path", type=str, default=config.RAW_DATASET_PATH, help="Path to the source JSON file.")
    parser.add_argument("--train_path", type=str, default=config.TRAIN_SET_PATH, help="Path to save the final training set.")
    parser.add_argument("--test_path", type=str, default=config.TEST_SET_PATH, help="Path to save the final test set.")
    parser.add_argument("--train_size", type=int, default=config.TRAIN_SIZE, help="Desired size of the training set.")
    parser.add_argument("--test_size", type=int, default=config.TEST_SIZE, help="Desired size of the test set.")
    parser.add_argument("--random_state", type=int, default=config.RANDOM_SEED, help="Random state for reproducibility.")
    args = parser.parse_args()
    create_final_datasets(
        args.source_path,
        args.train_path,
        args.test_path,
        args.train_size,
        args.test_size,
        args.random_state
    )
