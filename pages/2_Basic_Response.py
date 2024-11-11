import streamlit as st
import torch
from app import tokenizer, model, format_prompt, log_conversation, device

st.header("Basic Response")

# Retrieve system prompt from session state
system_prompt = st.session_state.get('system_prompt', "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.")
user_input = st.text_input("You: ", "")

def generate_response_basic(system_prompt, user_message, context=None):
    generation_settings = st.session_state['generation_settings']  # Retrieve settings

    formatted_prompt = context + format_prompt(system_prompt, user_message) if context else format_prompt(system_prompt, user_message)
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs,
        max_length=generation_settings['max_length'],
        temperature=generation_settings['temperature'],
        repetition_penalty=generation_settings['repetition_penalty'],
        no_repeat_ngram_size=generation_settings['no_repeat_ngram_size'],
        pad_token_id=tokenizer.eos_token_id,
        do_sample=generation_settings['enable_sampling']  # Use sampling based on checkbox
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if st.button("Send"):
    if user_input:
        conversation_history = ""
        if st.session_state.context_enabled:
            conversation_history = format_prompt(system_prompt, user_input)

        response_basic = generate_response_basic(system_prompt, user_input, conversation_history)
        st.text_area("Basic LLM Response:", value=response_basic, height=200)

        log_conversation({"user_input": user_input, "response_type": "Basic LLM", "response": response_basic})