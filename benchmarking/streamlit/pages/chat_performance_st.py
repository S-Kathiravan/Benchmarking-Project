import sys
import streamlit as st
import warnings
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("../")
from benchmarking.src.llmperf import common_metrics
from benchmarking.src.chat_performance_evaluation import ChatPerformanceEvaluator
from benchmarking.streamlit.app import LLM_API_OPTIONS

warnings.filterwarnings("ignore")


def _get_params() -> dict:
    """Get LLM params

    Returns:
        dict: returns dictionary with LLM params
    """
    params = {
        "max_tokens_to_generate": st.session_state.max_tokens_to_generate,
    }
    return params


def _parse_llm_response(llm: ChatPerformanceEvaluator, prompt: str) -> dict:
    """Parses LLM output to a dictionary with necessary performance metrics and completion

    Args:
        llm (ChatPerformanceEvaluator): Chat performance evaluation object
        prompt (str): user's prompt text

    Returns:
        dict: dictionary with performance metrics and completion text
    """
    llm_output = llm.generate(prompt=prompt)
    response = {
        "completion": llm_output[1],
        "time_to_first_token": float(llm_output[0].get(common_metrics.TTFT, 0.0)) if llm_output[0].get(
            common_metrics.TTFT) is not None else 0.0,
        "latency": float(llm_output[0].get(common_metrics.E2E_LAT, 0.0)) if llm_output[0].get(
            common_metrics.E2E_LAT) is not None else 0.0,
        "throughput": float(llm_output[0].get(common_metrics.REQ_OUTPUT_THROUGHPUT, 0.0)) if llm_output[0].get(
            common_metrics.REQ_OUTPUT_THROUGHPUT) is not None else 0.0,
    }
    return response


def _initialize_session_variables():
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "perf_metrics_history" not in st.session_state:
        st.session_state.perf_metrics_history = []
    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "llm_api" not in st.session_state:
        st.session_state.llm_api = None
    if "chat_disabled" not in st.session_state:
        st.session_state.chat_disabled = True

    # Initialize llm params
    if "max_tokens_to_generate" not in st.session_state:
        st.session_state.max_tokens_to_generate = 1024  # Default value for max tokens


def main():
    st.set_page_config(
        page_title="AI Starter Kit",
        page_icon="https://sambanova.ai/hubfs/logotype_sambanova_orange.png",
    )

    _initialize_session_variables()

    st.title(":orange[Fraud Detection and Evaluating Models]")
    st.markdown(
        "Provide transaction details to check if it's fraudulent."
    )

    with st.sidebar:
        st.title("Transaction Details")

        # Transaction details input fields
        transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "DEBIT", "CASH_OUT"])
        transaction_amount = st.number_input("Transaction Amount", min_value=0.0)
        nameOrig = st.text_input("Sender's Name (nameOrig)")
        oldbalanceOrg = st.number_input("Sender's Old Balance (oldbalanceOrg)", min_value=0.0)
        newbalanceOrig = st.number_input("Sender's New Balance (newbalanceOrig)", min_value=0.0)
        nameDest = st.text_input("Receiver's Name (nameDest)")
        oldbalanceDest = st.number_input("Receiver's Old Balance (oldbalanceDest)", min_value=0.0)
        newbalanceDest = st.number_input("Receiver's New Balance (newbalanceDest)", min_value=0.0)

        # Dropdown to select geographical location
        countries = sorted([
            "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia",
            "Australia", "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium",
            "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria",
            "Burkina Faso", "Burundi", "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic",
            "Chad", "Chile", "China", "Colombia", "Comoros", "Congo (Congo-Brazzaville)", "Costa Rica", "Croatia",
            "Cuba", "Cyprus", "Czechia (Czech Republic)", "Democratic Republic of the Congo", "Denmark", "Djibouti",
            "Dominica", "Dominican Republic", "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea",
            "Estonia", "Eswatini (fmr. Swaziland)", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia",
            "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
            "Haiti", "Honduras", "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel",
            "Italy", "Jamaica", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan",
            "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
            "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania",
            "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique",
            "Myanmar (formerly Burma)", "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua",
            "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway", "Oman", "Pakistan", "Palau", "Panama",
            "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar", "Romania",
            "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa",
            "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone",
            "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan",
            "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria", "Taiwan", "Tajikistan",
            "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey",
            "Turkmenistan", "Tuvalu", "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States",
            "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia", "Zimbabwe"
        ])
        location = st.selectbox("Geographical Location", countries)

        # Dropdown to specify if the transaction is fraud or not
        actual_fraud_status = st.selectbox("Is this transaction actually fraudulent?", ["fraud", "not fraud"])

        # LLM API type selection
        llm_api_selected = st.session_state.llm_api = st.selectbox("API type", options=LLM_API_OPTIONS)
        st.session_state.max_tokens_to_generate = st.number_input("Max tokens to generate", min_value=50,
                                                                  max_value=2048, value=250, step=1)

    # Button to run fraud detection
    if st.button("Check for Fraud"):
        with st.spinner("Checking transaction..."):

            # Prepare transaction data
            transaction_data = {
                "type": transaction_type,
                "amount": transaction_amount,
                "nameOrig": nameOrig,
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "nameDest": nameDest,
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "location": location,  # Include location in the prompt
            }

            # Create prompt for the LLM
            prompt = (
                f"Given the following transaction details:\n"
                f"Transaction Type: {transaction_data['type']}\n"
                f"Amount: {transaction_data['amount']}\n"
                f"Sender's Name: {transaction_data['nameOrig']}\n"
                f"Sender's Old Balance: {transaction_data['oldbalanceOrg']}\n"
                f"Sender's New Balance: {transaction_data['newbalanceOrig']}\n"
                f"Receiver's Name: {transaction_data['nameDest']}\n"
                f"Receiver's Old Balance: {transaction_data['oldbalanceDest']}\n"
                f"Receiver's New Balance: {transaction_data['newbalanceDest']}\n"
                f"Geographical Location: {transaction_data['location']}\n"
                f"Is this transaction fraudulent? Answer with 'yes' or 'no'."
            )

            # Set up LLMs for three models
            llm_models = ["llama3-405b", "llama3-70b", "llama3-8b"]
            results = {}

            for model in llm_models:
                llm_instance = ChatPerformanceEvaluator(model_name=model, llm_api=llm_api_selected,
                                                        params=_get_params())
                response = _parse_llm_response(llm_instance, prompt)
                results[model] = response

            # Check if all models predicted incorrectly
            all_models_incorrect = all(
                "fraud" if "yes" in results[model]['completion'].lower() else "not fraud" != actual_fraud_status
                for model in llm_models
            )

            if all_models_incorrect:
                st.warning(
                    "🚨 All models predicted incorrectly. Please try again with correct details."
                )
            else:
                # Display predictions at the top
                st.subheader("Model Predictions")

                for model in llm_models:
                    result = results[model]['completion'].strip().lower()
                    if "yes" in result:
                        st.error(f"🚨 Model {model} marks this transaction as **fraudulent**!")
                    elif "no" in result:
                        st.success(f"✅ Model {model} marks this transaction as **not fraudulent**.")
                    else:
                        st.warning(f"⚠️ Model {model} could not determine the fraud status confidently.")

                # Display the performance metrics plots
                st.subheader("Performance Metrics")

                for i, metric in enumerate(['time_to_first_token', 'latency', 'throughput']):
                    fig, ax = plt.subplots()
                    x = llm_models
                    values = [results[model][metric] for model in llm_models]
                    bars = ax.bar(x, values, color=plt.get_cmap('tab10')(i / 3.0))

                    # Add some text for labels, title, and custom x-axis tick labels, etc.
                    ax.set_xlabel('Models')
                    ax.set_ylabel('Values')
                    ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
                    st.pyplot(fig)

                # Compare models and generate summary
                _compare_models(results, llm_models, actual_fraud_status)


def _compare_models(results: dict, llm_models: list, actual_fraud_status: str):
    """Compares the performance of models and generates a summary, considering the actual fraud status"""

    # Track best model and metrics
    best_model = None
    best_metrics = {
        'time_to_first_token': float('inf'),
        'latency': float('inf'),
        'throughput': float('-inf'),
    }

    accuracy = {}
    correct_predictions = []
    incorrect_predictions = []

    for model in llm_models:
        result = results[model]
        prediction = "fraud" if "yes" in result['completion'].lower() else "not fraud"

        # Evaluate accuracy
        accuracy[model] = prediction == actual_fraud_status
        if accuracy[model]:
            correct_predictions.append(model)
        else:
            incorrect_predictions.append(model)

    # Find the best model among correct predictions based on evaluation order
    for model in correct_predictions:
        result = results[model]
        if result['time_to_first_token'] < best_metrics['time_to_first_token']:
            best_metrics['time_to_first_token'] = result['time_to_first_token']
            best_model = model
        elif result['time_to_first_token'] == best_metrics['time_to_first_token']:
            if result['latency'] < best_metrics['latency']:
                best_metrics['latency'] = result['latency']
                best_model = model
            elif result['latency'] == best_metrics['latency']:
                if result['throughput'] > best_metrics['throughput']:
                    best_metrics['throughput'] = result['throughput']
                    best_model = model

    # Generate summary
    description = (
        f"The best model in terms of overall performance is **{best_model}**.\n"
        f"- Lowest Time to First Token: {results[best_model]['time_to_first_token']} seconds\n"
        f"- Lowest Latency: {results[best_model]['latency']} seconds\n"
        f"- Highest Throughput: {results[best_model]['throughput']} tokens/second\n"
        "\nAccuracy Comparison:\n"
        f"- Correct Predictions: {', '.join(correct_predictions) if correct_predictions else 'None'}\n"
        f"- Incorrect Predictions: {', '.join(incorrect_predictions) if incorrect_predictions else 'None'}"
    )

    st.subheader("Model Comparison Summary")
    st.markdown(description)


if __name__ == "__main__":
    main()
