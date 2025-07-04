import json
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Configuration ---
import sys
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

# --- Function to Load Data ---
def load_data(file_path):
    """Loads data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
    return None

# --- Function to Extract Operations ---
def extract_operations(program_string):
    """Extracts all operation names from a program string."""
    if not isinstance(program_string, str):
        return []
    # Regex to find words followed by an opening parenthesis
    return re.findall(r'([a-zA-Z_]+)\(', program_string)

# --- Main Analysis and Visualization ---
def analyze_and_visualize(data, output_filename):
    """Analyzes program operations and visualizes their frequencies."""

    all_operations = []
    for split in data.values():
        for item in split:
            turn_programs = item.get('dialogue', {}).get('turn_program', [])
            for prog_str in turn_programs:
                operations = extract_operations(prog_str)
                all_operations.extend(operations)

    if not all_operations:
        print("No operations found in the dataset.")
        return

    op_counts = Counter(all_operations)
    
    # Sort by frequency for plotting
    sorted_ops = op_counts.most_common()
    ops, counts = zip(*sorted_ops)

    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x=list(counts), y=list(ops), hue=list(ops), palette="viridis", legend=False)
    plt.xlabel("Frequency")
    plt.ylabel("Operation Type")
    plt.title("Distribution of Program Operations in ConvFinQA Dataset")

    # Add frequency labels to each bar
    for i, v in enumerate(counts):
        ax.text(v, i, f"  {v}", color='black', va='center', fontweight='bold')

    plt.tight_layout()

    # Ensure the output directory exists
    output_dir = os.path.dirname(output_filename)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        plt.savefig(output_filename)
        print(f"Plot saved to {output_filename}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    
    plt.clf()
    plt.close()

    print("\n--- Operation Frequencies ---")
    for op, count in sorted_ops:
        print(f"- {op}: {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize the distribution of program operations.")
    parser.add_argument("--dataset_path", type=str, default=config.RAW_DATASET_PATH, help="Path to the raw ConvFinQA dataset.")
    parser.add_argument("--output_filename", type=str, default=config.FIGURES_DIR / "operations_distribution.png", help="Path to save the output plot.")
    args = parser.parse_args()

    dataset = load_data(args.dataset_path)
    if dataset:
        analyze_and_visualize(dataset, args.output_filename)
