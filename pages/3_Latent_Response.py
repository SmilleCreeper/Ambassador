import streamlit as st
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor, LogitsProcessorList
from app import tokenizer, model, format_prompt, log_conversation, device

st.header("Latent Response")

# Retrieve system prompt from session state
system_prompt = st.session_state.get('system_prompt', "You are a helpful, smart, kind, and efficient AI assistant. You always fulfill the user's requests to the best of your ability.")
user_input = st.text_input("You: ", "")

latent_vector = torch.randn(4096, device=device)

class DynamicLatentLogitsProcessor(LogitsProcessor):
    def __init__(self, latent_vector, model):
        self.latent_vector = latent_vector
        self.model = model
    
    def __call__(self, input_ids, logits):
        token_embeddings = self.model.get_input_embeddings().weight.to(device)
        token_similarities = F.cosine_similarity(token_embeddings, self.latent_vector.unsqueeze(0), dim=-1)
        return logits + token_similarities * 5

def generate_response_with_latent(system_prompt, user_message, latent_vector, context=None):
    generation_settings = st.session_state['generation_settings']  # Retrieve settings

    formatted_prompt = context + format_prompt(system_prompt, user_message) if context else format_prompt(system_prompt, user_message)
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(device)
    logits_processor = DynamicLatentLogitsProcessor(latent_vector, model)

    outputs = model.generate(
        inputs,
        max_length=generation_settings['max_length'],
        logits_processor=LogitsProcessorList([logits_processor]),
        repetition_penalty=generation_settings['repetition_penalty'],
        no_repeat_ngram_size=generation_settings['no_repeat_ngram_size'],
        temperature=generation_settings['temperature'],
        do_sample=generation_settings['enable_sampling']  # Use sampling based on checkbox
    )

    generated_tokens = outputs[0]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)

if st.button("Send"):
    if user_input:
        conversation_history = ""
        if st.session_state.context_enabled:
            conversation_history = format_prompt(system_prompt, user_input)

        response_latent = generate_response_with_latent(system_prompt, user_input, latent_vector, conversation_history)
        st.text_area("Character (Inner World):", value=response_latent, height=200)

        log_conversation({"user_input": user_input, "response_type": "Inner World", "response": response_latent})