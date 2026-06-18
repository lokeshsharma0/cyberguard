import os
import gradio as gr
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# 1. Grab authentication token
hf_token = os.environ.get("HF_TOKEN")

# 2. Download directly from your training repo
# NOTE: If your trained model is in another repo, change "cyberguard" to "qwen-cyber-custom" or "qwen-cyber-expert"
model_path = hf_hub_download(
    repo_id="Lowkey333/cyberguard", 
    filename="qwen-cyber-q4.gguf",
    token=hf_token
)

# 3. Optimize execution configuration flags for CPU performance
llm = Llama(
    model_path=model_path, 
    n_ctx=512,
    n_threads=4,        # Core threading optimization to speed up CPU processing
    n_batch=16          # Reduces RAM latency pressure during text processing
)

def chat(system_prompt, history, user_message):
    # Pure direct prompt template with absolute zero filler
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
    
    output = llm(
        prompt,
        max_tokens=150,    # Lowering token ceiling limits CPU generation wait time
        temperature=0.1,   # Drops creative variance to lock onto fine-tuned responses
        top_p=0.8,
        stop=["<|im_end|>"]
    )
    
    return output["choices"][0]["text"].strip()

# Custom UI Layout Block
with gr.Blocks(title="Gurugram Police AI") as demo:
    gr.Markdown("# Gurugram Police AI - Cyberguard System")
    
    with gr.Row():
        with gr.Column():
            sys_input = gr.Textbox(
                label="System Alignment", 
                value="You are Cyberguard, an elite cybersecurity AI assistant trained exclusively for the Gurugram Police. Speak only using your custom defense datasets."
            )
            user_input = gr.Textbox(label="Enter Cybersecurity Query / Scenario")
            submit_btn = gr.Button("Submit Query", variant="primary")
        with gr.Column():
            output_box = gr.Textbox(label="Cyberguard System Analysis Output", interactive=False)
            
    submit_btn.click(
        fn=chat, 
        inputs=[sys_input, gr.State(""), user_input], 
        outputs=output_box
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
