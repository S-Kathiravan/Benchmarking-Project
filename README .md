Benchmarking
======================

<!-- TOC -->
<!-- /TOC -->

# Overview
The **Fraud Detection Model Benchmarking Kit** enables the evaluation of different LLM models for fraud detection tasks. Hosted on SambaStudio, it provides configurable connectors to interact with SambaNova Cloud and perform both synthetic and custom performance evaluations. Users can assess model predictions and metrics through an intuitive app interface and leverage bash scripts for detailed performance analysis. This tool aims to streamline the comparison and benchmarking of LLM models for fraud detection scenarios.

This sample is ready-to-use. We provide:

Instructions for setting up with the LLM models in SambaStudio and SambanNova Cloud.
Instructions for running the fraud detection and performance evaluation models out-of-the-box.
Instructions for customizing and experimenting with different parameters and prompts to suit your specific needs.
   
# Before you begin

To perform this setup, you must be a SambaNova customer with a SambaStudio account or have a SambaNova Cloud API key (more details in the following sections). You also have to set up your environment before you can run or customize the starter kit. 

_These steps assume a Mac/Linux/Unix shell environment. If using Windows, you will need to adjust some commands for navigating folders, activating virtual environments, etc._

## Clone this repository

Clone the starter kit repo.
```bash
git clone https://github.com/S-Kathiravan/Benchmarking-Project.git
```

## Set up the inference endpoint, and environment variables

### Setup SambaNova Cloud

The next step is to set up your environment variables to use one of the models available from SambaNova. If you're a current SambaNova customer, you can deploy your models with SambaNova Cloud. If you are not a SambaNova customer, you can self-service provision API endpoints using SambaNova Cloud.

- **SambaNova Cloud**: Please follow the instructions [here](../README.md#use-sambanova-cloud-option-1) for setting up your environment variables.


## Create the (virtual) environment
1. (Recommended) Create a virtual environment and activate it (python version 3.11 recommended): 
    ```bash
    python<version> -m venv <virtual-environment-name>
    source <virtual-environment-name>/bin/activate
    ```

2. Install the required dependencies:
    ```bash
    cd benchmarking # If not already in the benchmarking folder
    pip install -r requirements.txt
    ```

# Use the starter kit

When using the benchmarking starter kit, you have two options for running the program:

- [*GUI Option*](#gui-option): This option contains plots and configurations from a web browser.
- [*CLI Option*](#cli-option): This option allows you to run the program from the command line and provides more flexibility.

## GUI Option

The GUI for this starter kit uses Streamlit, a Python framework for building web applications. This method is useful for analyzing outputs in a graphical manner since the results are shown via plots in the UI.

### Deploy the starter kit GUI

Ensure you are in the `benchmarking` folder and run the following command:

```shell
streamlit run streamlit/app.py --browser.gatherUsageStats false 
```

**After deploying the starter kit, you will see the following user interface:**

![image](https://github.com/user-attachments/assets/fd91a6e7-18cd-4f7b-a1fe-e1ce39533856)


# Quickstart 
### Features

- **Transaction Details Input**: Users can enter various transaction details including:
  - Transaction Type (e.g., PAYMENT, TRANSFER, DEBIT, CASH_OUT)
  - Transaction Amount
  - Sender's Information (Name, Old Balance, New Balance)
  - Receiver's Information (Name, Old Balance, New Balance)
  - Geographical Location

- **LLM Selection**: Choose from multiple LLMs to evaluate the transaction. The models available for selection are:
  - `llama3-405b`
  - `llama3-70b`
  - `llama3-8b`

- **Fraud Detection**: Based on the transaction details and selected models, the application generates predictions indicating whether the transaction is fraudulent or not.

- **Performance Metrics**: For each model, the application provides detailed performance metrics:
  - **Time to First Token**: Measures the time taken by the model to generate the first token of its response.
  - **Latency**: Represents the overall delay experienced during the response generation.
  - **Throughput**: Indicates the rate at which the model processes tokens.

- **Model Comparison**: The application compares the performance of different models and highlights:
  - The best-performing model based on the lowest time to first token, lowest latency, and highest throughput.
  - Accuracy of each model in detecting fraud, showcasing which models correctly identified fraudulent transactions and which did not.

### How It Works

1. **Input Data**: 
   - **Transaction Details**: Fill in the form with specific transaction information. This includes selecting the type of transaction (e.g., PAYMENT, TRANSFER, DEBIT, CASH_OUT) and providing details such as the transaction amount. 
   - **Sender and Receiver Information**: Enter the names of the sender and receiver, along with their respective old and new balances. These details help the model understand the financial context of the transaction.
   - **Geographical Location**: Choose the geographical location from a comprehensive list of countries. This can influence the fraud detection process by adding contextual information about the transaction’s origin.
     
![image](https://github.com/user-attachments/assets/d1877c7c-3c18-4fbb-b2fb-0b1d53f69926)

![image](https://github.com/user-attachments/assets/4d3b20c1-8e31-40fd-8517-2365b541c435)


2. **Select Models**

  - **Model Evaluation**: The application runs all selected large language models (LLMs) simultaneously to assess the transaction data. Available models include `llama3-405b`, `llama3-70b`, and `llama3-8b`. Each model has different capacities and performance characteristics, impacting their ability to detect fraud. After clicking "Check for Fraud," results from all models will be provided, allowing for a comprehensive comparison of their performance.


3. **Generate Predictions**: 
   - **Initiate Fraud Check**: Click the button to start the fraud detection process. The application will send the transaction details to the selected models and wait for their responses.
   - **Model Processing**: Each LLM evaluates the transaction based on the input data and generates predictions. The models assess the likelihood of the transaction being fraudulent or not, using their respective algorithms and trained data.
   ![image](https://github.com/user-attachments/assets/f35094f5-3b03-45c8-a27e-972cd37f4397)


4. **View Results**:
   - **Model Predictions**: The application displays the predictions from each model, indicating whether the transaction is classified as fraudulent or not. These predictions help you understand how each model interprets the data and provides its assessment.
   - **Performance Metrics**: Visualize the performance of each model through graphical plots. These plots compare key metrics such as:
     - **Time to First Token**: The duration taken by each model to generate its first response token. Shorter times indicate faster processing.
![image](https://github.com/user-attachments/assets/0ac34496-e985-4278-b088-f6408a3b4068)


     - **Latency**: The total time taken from receiving the input to providing the final response. Lower latency reflects quicker responses.
![image](https://github.com/user-attachments/assets/61cff088-5530-4f51-ace3-165a268cb89c)


     - **Throughput**: The rate at which each model processes tokens, measured in tokens per second. Higher throughput indicates better efficiency in handling large volumes of data.
![image](https://github.com/user-attachments/assets/829f868d-57e8-4deb-a0f0-137bc743fb0d)



   - **Model Comparison Summary**: Review a detailed summary that highlights the performance of each model. This includes:
     - **Accuracy**: Information on which models correctly or incorrectly predicted the fraud status of the transaction.
     - **Best-Performing Model**: Identification of the top model based on the lowest time to first token, lowest latency, and highest throughput. This helps in determining the most efficient model for your needs.
![image](https://github.com/user-attachments/assets/6824c768-2a01-4293-9a8a-21a30bf927dc)


### Insights and Conclusions

This application offers a comprehensive evaluation of different LLMs for fraud detection, highlighting their strengths and weaknesses through detailed performance metrics. By analyzing model predictions and comparing key metrics like time to first token, latency, and throughput, users gain valuable insights into the effectiveness of each model. The summary helps in identifying the best-performing model and understanding its reliability in detecting fraudulent transactions.
