import streamlit as st
import torch
from app import tokenizer, model, format_prompt, log_conversation, device

st.header("Confusing Response")

# Retrieve system prompt from session state
system_prompt = st.session_state.get('system_prompt', "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.")
user_input = st.text_input("You: ", "")

def generate_response_confusing(system_prompt, user_message, context=None):
    generation_settings = st.session_state['generation_settings']  # Retrieve settings

    formatted_prompt = (context + format_prompt(system_prompt, user_message)) if context else format_prompt(system_prompt, user_message)
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)

    response = []
    generated_tokens = inputs

    for _ in range(generation_settings['max_length']):
        with torch.no_grad():
            outputs = model(generated_tokens)
            logits = outputs.logits[:, -1, :]

            probabilities = torch.softmax(logits, dim=-1)
            least_likely_token = torch.argmin(probabilities, dim=-1)

        generated_tokens = torch.cat([generated_tokens, least_likely_token.unsqueeze(0)], dim=1)
        response.append(least_likely_token.item())

        if least_likely_token == tokenizer.eos_token_id:
            break

    # Decode only the generated tokens after the initial input
    response_text = tokenizer.decode(response, skip_special_tokens=True).strip()
    return response_text

if st.button("Send"):
    if user_input:
        conversation_history = ""
        if st.session_state.context_enabled:
            conversation_history = format_prompt(system_prompt, user_input)

        # Generate and display response
        response_confusing = generate_response_confusing(system_prompt, user_input, conversation_history)
        st.text_area("Confusing LLM Response:", value=response_confusing, height=200)

        # Log the conversation
        log_conversation({"user_input": user_input, "response_type": "Confusing LLM", "response": response_confusing})
