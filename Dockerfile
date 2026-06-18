# Use a standard, universally accessible Python version
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for runtime
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /lib/apt/lists/*

# Force install the pre-compiled binary wheel for llama-cpp-python so it skips compiling entirely!
RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Install Gradio and huggingface_hub cleanly
RUN pip install --no-cache-dir gradio huggingface_hub

# Copy the rest of the app files
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
