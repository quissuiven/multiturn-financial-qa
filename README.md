# Multiturn Financial Question Answering

## Objective

This project fine-tunes and evaluates a conversational question-answering system on the ConvFinQA dataset to answer financial questions based on tables and text.

## Video Demos

Video demonstrations of the application using CLI or Streamlit can be found in the `demos` folder.

## System Design and Evaluation

A detailed explanation of the system design and performance can be found in `REPORT.pdf`.

## Project Structure

The project is organized with a clear separation of concerns, making it modular and maintainable.

```
.
├── app/
│   └── app.py              # Streamlit web application for the interactive demo.
│
├── data/
│   ├── processed/          # Processed datasets, including train and test sets.
│   └── raw/                # The original, unmodified dataset.
│
├── figures/                # Output directory for plots and visualizations.
│
├── notebooks/              # Jupyter notebooks for exploration and fine-tuning.
│
├── outputs/
│   ├── analysis/           # CSV files containing error analysis from evaluations.
│   └── predictions/        # JSON files with model predictions from inference scripts.
│
├── scripts/                # Standalone scripts for data visualization, processing, inference and evaluation.
│
├── src/                    # Core project package. Contains shared logic like database utilities
│   │                       # (db_utils.py), program evaluation (program_utils.py), and central
│   │                       # configuration (config.py).
│   ├── __init__.py
│   ├── config.py
│   ├── db_utils.py
│   ├── main.py
│   └── program_utils.py
│
├── demos/                  # Contains video demonstrations of the project.
│
├── Dockerfile              # Instructions to build the Docker image for deployment.
├── pyproject.toml          # Project metadata and dependencies for `uv`.
├── requirements.txt        # Dependency list for deployment platforms like Render.
└── REPORT.pdf              # The final report summarizing findings and methodology.
```

---

## Instructions for running the code locally

### Prerequisites
- Python 3.9+
- [UV environment manager](https://docs.astral.sh/uv/getting-started/installation/)

### Setup
1.  **Clone this repository.**
2.  **Install UV:**
    ```bash
    brew install uv
    ```
3.  **Set up the environment and install dependencies:**
    ```bash
    uv sync
    ```
4.  **Set up Environment Variables:**
    - Create a `.env` file in the root directory.
    - Add your secret keys to this file. For most scripts, you will need:
      ```
      OPENAI_API_KEY="your_openai_api_key"
      MONGODB_URI="your_mongodb_connection_string"
      ```
    - The application logic in `src/config.py` will automatically load this file in a local development environment.

### Running Python Scripts Locally

All scripts are designed to be run from the root of the project directory. They use paths and parameters from `src/config.py` as defaults, which can be overridden with command-line arguments if needed.

- **Visualize Operations Distribution in Dataset:**
  ```bash
  python3 scripts/visualize_operations_dist.py
  ```
- **Prepare Train and Test Sets:**
  ```bash
  python3 scripts/prepare_train_test_sets.py
  ```
- **Validate Datasets & Generate Plots:**
  ```bash
  python3 scripts/validate_train_test_sets.py
  ```
- **Convert Datasets for Fine-tuning:**
  ```bash
  python3 scripts/convert_datasets_for_finetuning.py
  ```
- **Run Baseline Model Inference:**
  ```bash
  python3 scripts/run_baseline_inference.py --llm openai
  ```
- **Run Fine-tuned Model Inference:**
  ```bash
  python3 scripts/run_finetuned_inference.py
  ```
- **Run Evaluation:**
  ```bash
  python3 scripts/run_evaluation.py --predictions_path outputs/predictions/your_prediction_file.json
  ```
- **Load Data to MongoDB:**
  ```bash
  python3 scripts/load_data_to_mongodb.py --source_path data/raw/convfinqa_dataset.json
  ```

### Setting up and Using the CLI

To use the interactive CLI, you must first load the financial data into your MongoDB instance.

1.  **Set up a MongoDB Instance:**
    - You can use a free tier on [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register).
    - After creating your cluster, make sure to get the connection string (URI) and add it to your `.env` file.
    - In the "Network Access" tab of your Atlas dashboard, add your current IP address to the access list.

2.  **Load Data into MongoDB:**
    - Run the following script to load the entire raw dataset into your database. This script will clear the collection before inserting the new data.
      ```bash
      python3 scripts/load_data_to_mongodb.py --source_path data/raw/convfinqa_dataset.json
      ```

3.  **Chat with the CLI:**
    - Now you can use the `uv run main chat` command to interact with the model.
      ```bash
      uv run main chat "Single_JKHY/2009/page_28.pdf-3"
      ```
      - The final argument (`"Single_JKHY/2009/page_28.pdf-3"`) is the `record_id` for a specific financial document in the dataset. You can find `record_id` examples in the `data/raw/convfinqa_dataset.json` file.

      - **Note on Model Access:** The CLI defaults to using a specific fine-tuned model that is not public. To use your own model, you must update the `FINETUNED_MODEL_NAME` variable in `src/config.py` with your own model name from OpenAI.

### Interactive Demo (Streamlit App)

To run the interactive demo, you must add a password to your `.env` file:
```
APP_PASSWORD="a_password_for_the_local_streamlit_app"
```

You can run the app by using the following command:
```bash
streamlit run app/app.py
```

---

## Deployment to Render

The application can be deployed to Render using the provided `Dockerfile`.

1.  **Build and Push the Docker Image:**
    ```bash
    # Build for Render's architecture (linux/amd64)
    docker build --platform linux/amd64 -t your-dockerhub-username/convfinqa-app .
    
    # Log in to Docker Hub
    docker login
    
    # Push to Docker Hub
    docker push your-dockerhub-username/convfinqa-app
    ```
2.  **Create a New Web Service on Render:**
    - From your Render dashboard, click **"New +"** and select **"Web Service"**.
    - Choose to **"Deploy an existing image from a registry"**.
    - For the **Image Path**, enter `your-dockerhub-username/convfinqa-app:latest`.

3.  **Add Environment Variables:**
    - In the Render dashboard, go to the "Environment" section.
    - Add the following secrets:
      - `OPENAI_API_KEY`
      - `MONGODB_URI`
      - `APP_PASSWORD` (use a strong, secret password for the live demo)
      - `ENVIRONMENT` (set this to `production` to disable local `.env` loading)

4.  **Database Access:**
    - In your MongoDB Atlas dashboard, navigate to **"Network Access"**.
    - Ensure you have an entry that allows connections from anywhere (`0.0.0.0/0`). This is necessary for Render's servers to reach your database.
