FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-compiled binary for llama-cpp-python to skip compilation entirely
RUN pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
RUN pip install --no-cache-dir gradio huggingface_hub

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
