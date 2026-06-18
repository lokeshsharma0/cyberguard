iface = gr.Interface(
    fn=chat,
    inputs=[
        gr.Textbox(label="System Prompt", value="You are a helpful cybersecurity assistant named Cyberguard."),
        gr.Textbox(label="History (Optional)"),
        gr.Textbox(label="User Message")
    ],
    outputs=gr.Textbox(label="Response"),
    title="CYBERGUARD AI",
    flagging_mode="never"  # Updated from allow_flagging="never"
)

iface.launch(server_name="0.0.0.0", server_port=7860)
