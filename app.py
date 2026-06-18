import os
import gradio as gr
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

hf_token = os.environ.get("HF_TOKEN")

model_path = hf_hub_download(
    repo_id="Lowkey333/cyberguard-weights",
    filename="qwen-cyber-q4.gguf",
    token=hf_token
)

llm = Llama(
    model_path=model_path,
    n_ctx=512,
    n_threads=4,
    n_batch=16
)

def chat(system_prompt, user_message):
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    output = llm(
        prompt,
        max_tokens=150,
        temperature=0.1,
        top_p=0.8,
        stop=["<|im_end|>"]
    )
    return output["choices"][0]["text"].strip()

with gr.Blocks(title="Gurugram Police AI", theme=gr.themes.Base()) as demo:
    gr.Markdown("# Gurugram Police AI — Cyberguard System")
    with gr.Row():
        with gr.Column():
            sys_input = gr.Textbox(
                label="System Alignment",
                value="You are Cyberguard, an elite cybersecurity AI assistant trained exclusively for the Gurugram Police. Speak only using your custom defense datasets."
