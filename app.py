import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. This reads the secret you just saved in the Space settings panel
hf_token = os.environ.get("HF_TOKEN")

# 2. Make sure spelling matches your model repo exactly
MODEL_ID = "Lowkey333/cyberguard" 

# 3. Load with the token parameter included
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, token=hf_token)

# 4. Define Chat Logic
def chat(system_prompt, history, user_message):
    prompt = f"{system_prompt}\n\nUser: {user_message}\nAssistant:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's reply
    reply = response.split("Assistant:")[-1].strip()
    return reply

# 5. Launch Gradio Interface
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
