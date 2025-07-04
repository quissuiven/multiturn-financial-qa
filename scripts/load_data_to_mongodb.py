import json
import argparse
import sys
import os
from tqdm import tqdm

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.program_utils import dict_to_markdown_table
from src.db_utils import bulk_insert_data
from src import config

def main():
    """
    Main function to load, process, and upload data to MongoDB.
    """
    parser = argparse.ArgumentParser(description="Load and process financial data into MongoDB.")
    parser.add_argument("--source_path", type=str, default=config.TEST_SET_PATH, help="Path to the source JSON file.")
    parser.add_argument("--collection_name", type=str, default=config.MONGODB_COLLECTION, help="Name of the MongoDB collection.")
    args = parser.parse_args()

    # --- 1. Load and Process Data ---
    try:
        with open(args.source_path, 'r', encoding='utf-8') as f:
            source_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Source file not found at {args.source_path}")
        return

    # Handle both list-of-records and dict-of-lists-of-records formats
    if isinstance(source_data, dict):
        all_samples = [item for split in source_data.values() for item in split]
    else:
        all_samples = source_data
    
    print(f"Processing {len(all_samples)} samples from {args.source_path}...")
    
    documents_to_insert = []
    for sample in tqdm(all_samples, desc="Preparing documents"):
        doc = sample.get('doc', {})
        table_json = doc.get('table', {})
        
        transformed_doc = {
            "id": sample.get("id"),
            "doc": {
                "pre_text": doc.get("pre_text"),
                "post_text": doc.get("post_text"),
                "table": table_json,
                "table_markdown": dict_to_markdown_table(table_json)
            }
        }
        documents_to_insert.append(transformed_doc)

    # --- 2. Insert into MongoDB using the utility function ---
    success = bulk_insert_data(documents_to_insert, args.collection_name)
    
    if success:
        print("\nData loading process completed successfully.")
    else:
        print("\nData loading process failed.")

if __name__ == '__main__':
    main()
