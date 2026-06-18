import gradio as gr
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

# Read the built-in token that Hugging Face Spaces automatically provide
hf_token = os.environ.get("HF_TOKEN")

model = AutoModelForCausalLM.from_pretrained("Lowkey333/cyberguard", token=hf_token)
tokenizer = AutoTokenizer.from_pretrained("Lowkey333/cyberguard", token=hf_token)

# Replace with your model path/name
MODEL_NAME = "cyberguard"

tokenizer = AutoTokenizer.from_pretrained(cyberguard)
model = AutoModelForCausalLM.from_pretrained(cyberguard)

def chat(system_prompt, history, user_message):
    prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the assistant's reply
    reply = response.split("Assistant:")[-1].strip()
    return reply

iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="System Prompt"),
        gr.Textbox(label="History"),
        gr.Textbox(label="User Message")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Gurugram Police AI",
    allow_flagging="never"
)

iface.launch()
