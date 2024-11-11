import streamlit as st
import torch
import random
import json
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
import torch.nn.functional as F
import os

# Set up the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model and Tokenizer Loading
model_name = "meta-llama/Llama-3.1-8B-Instruct"

@st.cache_resource
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model

# Use session state as a backup check to prevent loading twice
if 'model_loaded' not in st.session_state:
    # Load the tokenizer and model only once, then store in session state
    tokenizer, model = load_model_and_tokenizer(model_name)
    st.session_state['tokenizer'] = tokenizer
    st.session_state['model'] = model
    st.session_state['model_loaded'] = True  # Set flag indicating model has been loaded
else:
    # If already loaded, retrieve from session state
    tokenizer = st.session_state['tokenizer']
    model = st.session_state['model']

# Utility functions (seed setting, prompt formatting, conversation logging, etc.)
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)

def format_prompt(system_prompt, user_message, assistant_message=None):
    prompt = (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
              f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
              f"{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    if assistant_message:
        prompt += assistant_message
    return prompt

def log_conversation(conversation_data, log_file='logs/conversation_log.json'):
    # Define the absolute path based on the script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(script_dir, log_file)
    print(f"Log file will be saved to: {log_file_path}")

    # Ensure the directory for the log file exists
    log_dir = os.path.dirname(log_file_path)
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir)
            print(f"Directory '{log_dir}' created for log file.")
        except Exception as e:
            print(f"Failed to create directory '{log_dir}': {e}")
            return

    # Load existing logs or initialize as empty list
    logs = []
    if os.path.exists(log_file_path):
        try:
            with open(log_file_path, 'r', encoding="utf-8") as f:
                logs = json.load(f)
            print("Successfully loaded existing log data.")
        except json.JSONDecodeError:
            print(f"Warning: Log file '{log_file_path}' is corrupt or empty; starting fresh.")
        except Exception as e:
            print(f"Failed to load log file '{log_file_path}': {e}")
            return

    # Append new conversation data
    logs.append(conversation_data)
    print("Conversation data appended to log.")

    # Save updated logs back to the file
    try:
        with open(log_file_path, 'w', encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)
        print(f"Conversation data successfully saved to '{log_file_path}'.")
    except Exception as e:
        print(f"Failed to write to log file '{log_file_path}': {e}")

st.title("Latent-Driven AI Conversation with Context Memory")
st.write("Welcome! Use the sidebar to navigate between Settings, Basic Response, and Latent Response pages.")