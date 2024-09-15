import sys
import pandas as pd
import streamlit as st
from typing import List
import matplotlib.pyplot as plt
from st_pages import Page, show_pages
from dotenv import load_dotenv
import warnings

# Append necessary paths to system
sys.path.append("../")
sys.path.append("./src")
sys.path.append("./streamlit")

# Import custom modules
from benchmarking.src.performance_evaluation import SyntheticPerformanceEvaluator
from streamlit_utils import plot_client_vs_server_barplots, plot_dataframe_summary

# Suppress warnings
warnings.filterwarnings("ignore")

# Define constants
LLM_API_OPTIONS = ["sncloud"]


# Cache for data initialization
@st.cache_data
def _init():
    load_dotenv("../.env", override=True)


# Initialize session state variables
def _initialize_session_variables():
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "input_tokens" not in st.session_state:
        st.session_state.input_tokens = None
    if "output_tokens" not in st.session_state:
        st.session_state.output_tokens = None
    if "number_requests" not in st.session_state:
        st.session_state.number_requests = None
    if "number_concurrent_workers" not in st.session_state:
        st.session_state.number_concurrent_workers = None
    if "timeout" not in st.session_state:
        st.session_state.timeout = None
    if "llm_api" not in st.session_state:
        st.session_state.llm_api = None


# Function to run performance evaluation
def _run_performance_evaluation() -> pd.DataFrame:
    """Runs the performance evaluation process for different number of workers.
    Returns:
        pd.DataFrame: Dataframe with metrics for each number of workers.
    """
    results_path = "./data/results/llmperf"

    # Call benchmarking process
    performance_evaluator = SyntheticPerformanceEvaluator(
        model_name=st.session_state.llm,
        results_dir=results_path,
        num_workers=st.session_state.number_concurrent_workers,
        timeout=st.session_state.timeout,
        llm_api=st.session_state.llm_api,
    )

    performance_evaluator.run_benchmark(
        num_input_tokens=st.session_state.input_tokens,
        num_output_tokens=st.session_state.output_tokens,
        num_requests=st.session_state.number_requests,
        sampling_params={},
    )

    # Read generated json and output formatted results
    df_user = pd.read_json(performance_evaluator.individual_responses_file_path)
    df_user["concurrent_user"] = st.session_state.number_concurrent_workers
    valid_df = df_user[(df_user["error_code"] != "")]

    # For non-batching endpoints, batch_size_used will be 1
    if valid_df["batch_size_used"].isnull().all():
        valid_df["batch_size_used"] = 1

    return valid_df


# Main function to run the app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    # **Check Page Paths**: Ensure the paths provided here are valid and correct.
    show_pages(
        [
           # Page("streamlit/app.py", "Synthetic Performance Evaluation"),  # Valid page here
            #Page("streamlit/pages/custom_performance_eval_st.py", "Custom Performance Evaluation"), # Custom page
            # Add any other valid pages that you may need
        ]
    )

    # Initialize environment variables and session state
    _init()
    _initialize_session_variables()

    # Sidebar configuration
    with st.sidebar:
        st.title("Configuration")
        st.markdown("**Modify the following parameters before running the process**")

        # Sidebar input for model name
        llm_model = st.text_input(
            "Model Name",
            value="llama3-70b",
            help="Look at your model card in SambaStudio and introduce the same name of the model/expert here.",
        )
        st.session_state.llm = llm_model

        # Sidebar input for LLM API
        st.session_state.llm_api = st.selectbox("API type", options=LLM_API_OPTIONS)

        # Sidebar input for number of tokens
        st.session_state.input_tokens = st.number_input(
            "Number of input tokens", min_value=50, max_value=2000, value=1000, step=1
        )
        st.session_state.output_tokens = st.number_input(
            "Number of output tokens", min_value=50, max_value=2000, value=1000, step=1
        )

        # Sidebar input for number of requests and concurrent workers
        st.session_state.number_requests = st.number_input(
            "Number of total requests", min_value=10, max_value=1000, value=32, step=1
        )
        st.session_state.number_concurrent_workers = st.number_input(
            "Number of concurrent workers", min_value=1, max_value=100, value=1, step=1
        )

        # Sidebar input for timeout
        st.session_state.timeout = st.number_input(
            "Timeout", min_value=60, max_value=1800, value=600, step=1
        )

        # Button to initialize the application
        sidebar_option = st.sidebar.button("Initialize Application")

    # Only initialize the app, without running performance evaluation
    if sidebar_option:
        st.success("App initialized! Now you can proceed to trigger performance evaluation from the selected page.")
        # This will ensure app.py is loaded but doesn't run the evaluator directly.

