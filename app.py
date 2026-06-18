import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
import os

# 1. Authenticate with your token
hf_token = os.environ.get("HF_TOKEN")

# 2. Download the single GGUF file from your model repo securely
print("Downloading GGUF model file...")
model_path = hf_hub_download(
    repo_id="Lowkey333/cyberguard",
    filename="qwen-cyber-q4.gguf",
    token=hf_token
)

# 3. Initialize the Llama-cpp engine with the downloaded file
print("Loading model into memory...")
llm = Llama(model_path=model_path, n_ctx=512)

# 4. Define Chat Inference Logic
def chat(system_prompt, history, user_message):
    # Construct a clean chat prompt format for the model
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    # Generate text response
    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.7,
        stop=["<|im_end|>", "User:"]
    )
    
    # Extract the text string response
    reply = output["choices"][0]["text"].strip()
    return reply

# 5. Build and Launch the Interface
iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful cybersecurity assistant named Cyberguard."),
        gr.Textbox(label="History (Optional)"),
        gr.Textbox(label="User Message")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Gurugram Police AI",
    allow_flagging="never"
)

iface.launch()
