FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME="/usr/local/cuda"
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV PATH="${CUDA_HOME}/bin:$PATH"

# Install system dependencies and Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ca-certificates \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for Python 3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Install pip for Python 3.11
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

# Check Python version
RUN python3 --version

# Install uv for faster package installation
RUN python3 -m pip install uv

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install wheel first to avoid build issues
RUN python3 -m uv pip install wheel packaging psutil ninja

# Install PyTorch with CUDA 12.4 support
# First uninstall any existing PyTorch installation to avoid conflicts
RUN python3 -m uv pip uninstall --system torch torchvision torchaudio
# Install PyTorch with CUDA 12.4 support
RUN python3 -m uv pip install --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# Verify CUDA is available
RUN echo python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}');"

# Create a script to install flash-attn 
RUN python3 -m uv pip install --system flash-attn --no-build-isolation

# Install dependencies from requirements.txt using uv
RUN if [ -s requirements.txt ]; then \
    python3 -m uv pip install --system -r requirements.txt; \
    else \
    echo "Warning: requirements.txt is empty or missing"; \
    fi

# Set up Hugging Face cache directories with proper permissions
RUN mkdir -p /app/.cache/huggingface && \
    chmod -R 777 /app/.cache

# Set environment variables for Hugging Face
ENV HF_DATASETS_CACHE="/app/.cache/huggingface/datasets"
ENV HF_HOME="/app/.cache/huggingface"

# Create a helper script for Hugging Face authentication
RUN echo '#!/bin/bash\n\
if [ -z "$1" ]; then\n\
  echo "Please provide your Hugging Face token as an argument"\n\
  echo "Usage: ./hf_login.sh YOUR_HF_TOKEN"\n\
  exit 1\n\
fi\n\
\n\
echo "Logging in to Hugging Face with provided token..."\n\
python3 -c "from huggingface_hub import login; login(\"$1\")"\n\
echo "Login successful! You can now access private and gated datasets."\n\
' > /app/hf_login.sh && chmod +x /app/hf_login.sh

# Create a non-root user and set proper permissions
RUN useradd -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    chmod -R 755 /app

# Create a directory for user data and set permissions
RUN mkdir -p /app/data && \
    chown -R appuser:appuser /app/data && \
    chmod -R 777 /app/data

# Set up the entrypoint
ENTRYPOINT ["/bin/bash"] 