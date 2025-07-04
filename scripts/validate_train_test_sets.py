import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def analyze_and_plot_distributions(
    train_path,
    test_path,
    output_dir
):
    """
    Analyzes and plots the distribution of key features in the final train and test sets.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load and Process Data ---
    try:
        with open(train_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        with open(test_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not find dataset files. {e}")
        return

    def process_data(data, dataset_name):
        processed = []
        for item in data:
            features = item.get('features', {})
            dialogue = item.get('dialogue', {})
            programs_str = str(dialogue.get('turn_program', []))
            processed.append({
                'dataset': dataset_name,
                'has_type_2_question': features.get('has_type2_question', False),
                'num_dialogue_turns': features.get('num_dialogue_turns', 0),
                'has_exp': 'exp' in programs_str,
                'has_greater': 'greater' in programs_str
            })
        return pd.DataFrame(processed)

    train_df = process_data(train_data, 'Train Set')
    test_df = process_data(test_data, 'Test Set')
    combined_df = pd.concat([train_df, test_df])

    print("--- Data Overview ---")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    # --- 2. Plotting Functions ---
    def plot_distribution(column_name, title, xlabel, file_name, xticklabels=None):
        plt.figure(figsize=(12, 7))
        ax = sns.countplot(data=combined_df, x=column_name, hue='dataset', palette='viridis', hue_order=['Train Set', 'Test Set'])
        plt.title(f'Distribution of {title}', fontsize=16)
        plt.ylabel('Count', fontsize=12)
        plt.xlabel(xlabel, fontsize=12)
        
        if xticklabels:
            # Set fixed ticks to avoid UserWarning
            ax.set_xticks(range(len(xticklabels)))
            ax.set_xticklabels(xticklabels)
        
        # Add count labels to each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, file_name)
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")

    # --- 3. Generate Plots for General Features ---
    plot_distribution(
        'has_type_2_question', 
        'Question Types', 
        'Question Type',
        'question_type_dist.png',
        xticklabels=['Type 1', 'Type 2']
    )
    plot_distribution(
        'num_dialogue_turns', 
        'Number of Dialogue Turns', 
        'Number of Turns',
        'dialogue_turns_dist.png'
    )

    # --- 4. Generate Specific Plot for Rare Operations ---
    rare_op_counts = combined_df.groupby('dataset')[['has_exp', 'has_greater']].sum().reset_index()
    rare_op_melted = rare_op_counts.melt(id_vars='dataset', var_name='operation', value_name='count')
    rare_op_melted['operation'] = rare_op_melted['operation'].str.replace('has_', '')

    plt.figure(figsize=(10, 6))
    ax_rare = sns.barplot(data=rare_op_melted, x='operation', y='count', hue='dataset', palette='viridis', hue_order=['Train Set', 'Test Set'])
    plt.title('Distribution of Samples Containing Rare Operations', fontsize=16)
    plt.ylabel('Count of Samples', fontsize=12)
    plt.xlabel('Operation Type', fontsize=12)

    # Add count labels
    for p in ax_rare.patches:
        ax_rare.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'rare_op_dist.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    
    print("\nDistribution analysis complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and plot dataset distributions.")
    parser.add_argument("--train_path", type=str, default=config.TRAIN_SET_PATH, help="Path to the final training set JSON file.")
    parser.add_argument("--test_path", type=str, default=config.TEST_SET_PATH, help="Path to the final test set JSON file.")
    parser.add_argument("--output_dir", type=str, default=config.FIGURES_DIR, help="Directory to save the output plots.")
    args = parser.parse_args()
    analyze_and_plot_distributions(args.train_path, args.test_path, args.output_dir)
