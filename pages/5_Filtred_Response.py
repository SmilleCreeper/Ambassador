import streamlit as st
import torch
from app import tokenizer, model, format_prompt, log_conversation, device
import torch.nn.functional as F

st.header("Real-Time Filtered Response")

# Retrieve system prompt from session state
system_prompt = st.session_state.get(
    'system_prompt', 
    "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability."
)
user_input = st.text_input("You: ", "")

# Tokens to avoid (we will set their generation probabilities to zero)
avoid_tokens = ["!", "?", "!?", "?!"]

# Find token IDs corresponding to the symbols in avoid_tokens
avoid_token_ids = set(tokenizer.encode(" ".join(avoid_tokens), add_special_tokens=False))

def generate_response_real_time_with_filter(system_prompt, user_message, context=None):
    """Generate response token-by-token, setting probabilities of avoid_tokens to zero in real-time."""
    generation_settings = st.session_state['generation_settings']
    formatted_prompt = context + format_prompt(system_prompt, user_message) if context else format_prompt(system_prompt, user_message)
    
    # Encode initial input
    input_ids = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    generated_tokens = input_ids  # Start with initial prompt tokens
    
    response_text = ""

    for _ in range(generation_settings['max_length']):
        with torch.no_grad():
            outputs = model(generated_tokens)
            logits = outputs.logits[:, -1, :]

            # Set probabilities of avoid_tokens to zero
            for avoid_token_id in avoid_token_ids:
                logits[0, avoid_token_id] = float('-inf')  # Set probability to zero

            # Apply temperature and sampling settings
            if generation_settings['enable_sampling']:
                probabilities = F.softmax(logits / generation_settings['temperature'], dim=-1)
                next_token_id = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1)

        # Decode the selected token and append to response
        next_token_text = tokenizer.decode(next_token_id, skip_special_tokens=True)
        response_text += next_token_text
        
        # Update generated tokens with the new token
        generated_tokens = torch.cat([generated_tokens, next_token_id.unsqueeze(0)], dim=1)

        # Check for end-of-sequence token
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return response_text

if st.button("Send"):
    if user_input:
        conversation_history = ""
        if st.session_state.context_enabled:
            conversation_history = format_prompt(system_prompt, user_input)

        # Generate response with filtered tokens
        response_filtered = generate_response_real_time_with_filter(system_prompt, user_input, conversation_history)
        st.text_area("Real-Time Filtered Response:", value=response_filtered, height=200)

        log_conversation({"user_input": user_input, "response_type": "Real-Time Filtered", "response": response_filtered})