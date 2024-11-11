import streamlit as st
from app import set_seed

st.header("Settings")

# Set random seed for deterministic behavior
seed = st.number_input("Random Seed", value=42, step=1)
set_seed(seed)

# Context Memory and Sampling toggle in the same row
col1, col2 = st.columns(2)

with col1:
    context_enabled = st.checkbox("Enable Context Memory", value=True)

with col2:
    enable_sampling = st.checkbox("Enable Sampling", value=False)

# User-defined system prompt input
system_prompt = st.text_area("System Prompt", 
                            value="You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.", 
                            height=100)

# Save settings in session state
st.session_state.context_enabled = context_enabled
st.session_state['generation_settings'] = {
    "max_length": st.slider("Max Length", min_value=20, max_value=4096, value=256, step=10),
    "temperature": st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1),
    "repetition_penalty": st.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.5, step=0.1),
    "no_repeat_ngram_size": st.slider("No Repeat N-Gram Size", min_value=1, max_value=10, value=2),
    "enable_sampling": enable_sampling  # Save "Enable Sampling" state
}
st.session_state['system_prompt'] = system_prompt  # Save system prompt

st.write("Settings updated. Use the other pages to generate responses.")