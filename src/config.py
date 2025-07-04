import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file only in development environments
if os.getenv('ENVIRONMENT') != 'production':
    load_dotenv()

# Define the absolute path to the project's root directory
# This makes all paths relative to the project root, ensuring it works on any machine
ROOT_DIR = Path(__file__).parent.parent

# --- Data Paths ---
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Specific File Paths
RAW_DATASET_PATH = RAW_DATA_DIR / "convfinqa_dataset.json"
TRAIN_SET_PATH = PROCESSED_DATA_DIR / "final_train_set.json"
TEST_SET_PATH = PROCESSED_DATA_DIR / "final_test_set.json"
TRAIN_SET_JSONL_PATH = PROCESSED_DATA_DIR / "final_train_set.jsonl"
TEST_SET_JSONL_PATH = PROCESSED_DATA_DIR / "final_test_set.jsonl"

# --- Output Paths ---
OUTPUTS_DIR = ROOT_DIR / "outputs"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
FIGURES_DIR = ROOT_DIR / "figures"

# --- Model Settings ---
OPENAI_MODEL = "gpt-4.1-2025-04-14"
FINETUNED_OPENAI_MODEL = 'ft:gpt-4.1-mini-2025-04-14:tsgs::BnoieExO'
GEMINI_MODEL = "models/gemini-2.5-pro-preview-05-06"

# LLM Call Parameters
TEMPERATURE = 0.0
MAX_TOKENS = 200

# --- Train Test Split Parameters ---
TRAIN_SIZE = 1000
TEST_SIZE = 200
RANDOM_SEED = 42

# --- Database Settings ---
MONGODB_DATABASE = "convfinqa"
MONGODB_COLLECTION = "parent_docs"
