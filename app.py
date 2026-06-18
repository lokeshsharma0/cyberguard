import os
import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# 1. Authenticate using your token
hf_token = os.environ.get("HF_TOKEN")

# 2. Change the repo_id if your trained file is in custom or expert!
print("Downloading your custom trained qwen-cyber-q4.gguf file...")
model_path = hf_hub_download(
    repo_id="Lowkey333/cyberguard",  # Change to "Lowkey333/qwen-cyber-custom" if needed!
    filename="qwen-cyber-q4.gguf",
    token=hf_token
)
# 3. Initialize the pre-installed local engine with your file
print("Loading model weights into memory...")
llm = Llama(model_path=model_path, n_ctx=512)

# 4. Define Direct Chat Inference Logic
def chat(system_prompt, history, user_message):
    # Construct a clean ChatML string manually to match Qwen's exact structure
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_message}<|im_end|>\n"
    prompt += f"<|im_start|>assistant\n"
    
    # Run raw inference directly against your GGUF file layers
    output = llm(
        prompt,
        max_tokens=256,        # Keeps responses fast and memory consumption low
        temperature=0.1,       # Forces it to pick the most deterministic tokens from your training data
        top_p=0.9,             # Nucleus sampling boundary to filter out creative hallucinations
        stop=["<|im_end|>", "<|im_start|>"]  # Hard stops to ensure it doesn't loop or echo tokens
    )
    
    # Extract just the generated response text cleanly
    return output["choices"][0]["text"].strip()

# 5. Build and Launch the Interface cleanly
iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(
            label="System Prompt", 
            value="You are Cyberguard, a specialized cybersecurity AI trained to help secure systems. Answer accurately as an expert cyber specialist."
        ),
        gr.Textbox(label="History (Optional)"),
        gr.Textbox(label="User Message")
    ],
    outputs=gr.Textbox(label="Response"),
    title="Gurugram Police AI",
    flagging_mode="never"
)
iface.launch(server_name="0.0.0.0", server_port=7860)
