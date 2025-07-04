"""
Main typer app for ConvFinQA
"""
import os
import typer
from rich import print as rich_print
from .db_utils import get_record_by_id
from .program_utils import eval_program, program_tokenization
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- App Initialization ---
app = typer.Typer(
    name="main",
    help="A CLI application to chat with a fine-tuned model about financial reports.",
    add_completion=True,
    no_args_is_help=True,
)
# --- Global Variables & Setup ---
from . import config

@app.command()
def chat(
    record_id: str = typer.Argument(..., help="ID of the record to chat about (e.g., 'Single_Apple/2005/page_35.pdf-1')"),
) -> None:
    """Ask questions about a specific financial record stored in MongoDB."""
    
    # --- 1. Retrieve the record from MongoDB ---
    record = get_record_by_id(record_id)
    
    if not record:
        rich_print(f"[bold red]Error: Record with ID '{record_id}' not found in the database.[/bold red]")
        raise typer.Exit(code=1)
    
    rich_print(f"[green]Successfully loaded record: {record_id}[/green]")
    
    # --- 2. Prepare the initial context (System Prompt) ---
    doc = record.get('doc', {})
    system_prompt = (
        f"{doc.get('pre_text', '')}\n\n"
        f"TABLE:\n{doc.get('table_markdown', '')}\n\n"
        f"{doc.get('post_text', '')}"
    )
    
    # --- 3. Initialize the conversation ---
    llm = ChatOpenAI(model=config.FINETUNED_OPENAI_MODEL, temperature=config.TEMPERATURE)
    history = [SystemMessage(content=system_prompt)]
    
    rich_print("[bold yellow]Starting chat session. Type 'exit' or 'quit' to end.[/bold yellow]")

    while True:
        message = input(">>> ")
        if message.strip().lower() in {"exit", "quit"}:
            rich_print("[bold yellow]Ending chat session.[/bold yellow]")
            break

        # --- 4. Invoke the LLM with the full conversation history ---
        history.append(HumanMessage(content=message))
        
        try:
            # Get the predicted program string
            response = llm.invoke(history)
            program_str = response.content.strip()
            rich_print(f"[grey50]Predicted program: {program_str}[/grey50]")
            
            # Add the model's program to the history for the next turn
            history.append(AIMessage(content=program_str))

            # --- 5. Execute the program to get the final answer ---
            tokenized_prog = program_tokenization(program_str)
            _, final_answer = eval_program(tokenized_prog)
 
            rich_print(f"[blue][bold]Assistant:[/bold] {final_answer}[/blue]")

        except Exception as e:
            rich_print(f"[bold red]An error occurred: {e}[/bold red]")
            # Remove the last user message to allow them to try again
            history.pop()

@app.command()
def myfunc() -> None:
    """My hello world function"""
    rich_print("Hello World")

# This check is what allows 'uv run main' to work correctly
if __name__ == "__main__":
    app()
