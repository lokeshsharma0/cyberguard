# Use the official Hugging Face pre-built image that already has llama.cpp and python ready
FROM ghcr.io/huggingface/llama-cpp-python:main

# Set up a working directory
WORKDIR /app

# Copy the current directory files into the container
COPY . .

# Install Gradio and huggingface_hub cleanly (no compiling needed!)
RUN pip install --no-cache-dir gradio huggingface_hub

# Expose the standard port for Hugging Face Spaces
EXPOSE 7860

# Run your app script
CMD ["python", "app.py"]
