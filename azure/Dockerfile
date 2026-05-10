FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest

# Install pinned dependencies at image build time — not at job run time.
# This makes every training job start instantly with a tested, deterministic environment.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Suppress tokenizer parallelism warnings in forked processes.
ENV TOKENIZERS_PARALLELISM=false
