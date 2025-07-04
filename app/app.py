import streamlit as st
import sys
import os

# Add the project root to the Python path to allow for module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.db_utils import get_record_by_id
from src.program_utils import eval_program, program_tokenization
from src import config
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# --- Page Configuration ---
st.set_page_config(
    page_title="ConvFinQA Demo",
    page_icon="💰",
    layout="wide"
)

# --- Hide Streamlit UI Elements ---
st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# --- Password Protection ---
def check_password():
    """Returns `True` if the user has the correct password."""
    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Please enter the password to access the application.", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Please enter the password to access the application.", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

def password_entered():
    """Checks whether a password entered by the user is correct."""
    if st.session_state["password"] == os.environ.get("APP_PASSWORD"):
        st.session_state["password_correct"] = True
        del st.session_state["password"]  # don't store password
    else:
        st.session_state["password_correct"] = False

# --- Main App Logic ---
if check_password():
    st.title("💰 ConvFinQA Demo")
    st.write("A web interface to chat with a fine-tuned model about financial reports.")

    # --- State Management ---
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'record_loaded' not in st.session_state:
        st.session_state.record_loaded = False
    if 'record_id' not in st.session_state:
        st.session_state.record_id = ""

    # --- Sidebar for Record ID Input ---
    with st.sidebar:
        st.header("Configuration")
        record_id_input = st.text_input(
            "Enter a Record ID", 
            value="Single_JKHY/2009/page_28.pdf-3",
            help="Example: 'Single_JKHY/2009/page_28.pdf-3'"
        )

        if st.button("Load Record"):
            st.session_state.record_id = record_id_input
            with st.spinner(f"Loading record: {st.session_state.record_id}..."):
                record = get_record_by_id(st.session_state.record_id)
                if not record:
                    st.error(f"Error: Record with ID '{st.session_state.record_id}' not found.")
                    st.session_state.record_loaded = False
                else:
                    st.success(f"Successfully loaded record: {st.session_state.record_id}")
                    doc = record.get('doc', {})
                    system_prompt = (
                        f"{doc.get('pre_text', '')}\n\n"
                        f"TABLE:\n{doc.get('table_markdown', '')}\n\n"
                        f"{doc.get('post_text', '')}"
                    )
                    st.session_state.history = [SystemMessage(content=system_prompt)]
                    st.session_state.record_loaded = True
        
        st.markdown("---")
        if st.button("Clear Chat History"):
            st.session_state.history = []
            st.session_state.record_loaded = False
            st.session_state.record_id = ""
            st.rerun()

    # --- Chat Interface ---
    if not st.session_state.record_loaded:
        st.info("Please load a record using the sidebar to begin the chat.")
    else:
        for message in st.session_state.history:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)

        if prompt := st.chat_input("Ask a question about the financial record..."):
            with st.chat_message("user"):
                st.markdown(prompt)
            
            st.session_state.history.append(HumanMessage(content=prompt))

            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    try:
                        llm = ChatOpenAI(model=config.FINETUNED_OPENAI_MODEL, temperature=config.TEMPERATURE)
                        response = llm.invoke(st.session_state.history)
                        program_str = response.content.strip()
                        
                        st.session_state.history.append(AIMessage(content=program_str))

                        tokenized_prog = program_tokenization(program_str)
                        _, final_answer = eval_program(tokenized_prog)
                        
                        st.markdown(f"**Answer:** {final_answer}")
                        with st.expander("View Generated Program"):
                            st.code(program_str, language="text")

                    except Exception as e:
                        st.error(f"An error occurred: {e}")
