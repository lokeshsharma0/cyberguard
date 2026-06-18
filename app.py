import os
import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# 1. Authenticate using your token
hf_token = os.environ.get("HF_TOKEN")

# 2. Download your exact GGUF file from your model repo
print("Downloading your qwen-cyber-q4.gguf file...")
model_path = hf_hub_download(
    repo_id="Lowkey333/cyberguard",
    filename="qwen-cyber-q4.gguf",
    token=hf_token
)

# 3. Initialize the pre-installed local engine with your file
print("Loading model weights into memory...")
llm = Llama(model_path=model_path, n_ctx=512)

# 4. Define Chat Inference Logic
def chat(system_prompt, history, user_message):
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.7,
        stop=["<|im_end|>", "User:"]
    )
    return output["choices"][0]["text"].strip()

# 5. Build and Launch the Interface cleanly
iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful cybersecurity assistant named Cyberguard."),
        gr.Textbox(label="History (Optional)"),
        gr.Textbox(label="User Message")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Gurugram Police AI",
    flagging_mode="never"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
