# Chatbot Uncensored

A local GGUF-based chatbot running through `llama-cpp-python`, with optional GPU acceleration, persona control through `SOUL.md`, and RAG support using ChromaDB.

This project is designed to run locally inside WSL Ubuntu with CUDA acceleration for NVIDIA GPUs.

## Features

* Local GGUF model inference
* `llama-cpp-python` backend
* WSL Ubuntu support
* NVIDIA GPU acceleration
* RTX 5090 / Blackwell CUDA build support
* Persona loading from `SOUL.md`
* Optional RAG document retrieval with ChromaDB
* Local document indexing from `docs/`
* Startup scripts for Windows and WSL

## Project Structure

```text
chatbot-uncensored/
  chatbot.py
  ingestDocs.py
  SOUL.md
  start-chatbot.sh
  start-chatbot-windows.ps1
  README.md
  .gitignore
  docs/
    example.md
  models/
    NemoMix-Unleashed-12B-Q4_K_M.gguf
    embedding-model.gguf
  chromaDb/
  .venv-wsl/
```

## Requirements

* Windows 11
* WSL 2
* Ubuntu on WSL
* Python 3.12 inside WSL
* NVIDIA GPU
* NVIDIA Windows driver with WSL CUDA support
* CUDA Toolkit 12.9 inside WSL
* `llama-cpp-python`
* `chromadb`

## WSL Setup

Install Ubuntu through PowerShell:

```powershell
wsl --install -d Ubuntu
```

Inside Ubuntu:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential cmake ninja-build git
```

## Virtual Environment

From the project folder:

```bash
cd /mnt/c/Users/sully/Documents/projects/chatbot-uncensored
python3 -m venv .venv-wsl
source .venv-wsl/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## CUDA Setup

This project was tested with CUDA Toolkit 12.9 for RTX 5090 / Blackwell support.

Set CUDA paths inside WSL:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export CUDAToolkit_ROOT=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
```

Check CUDA:

```bash
nvidia-smi
which nvcc
nvcc --version
```

Expected `nvcc` path:

```text
/usr/local/cuda-12.9/bin/nvcc
```

## Installing llama-cpp-python with CUDA

For RTX 5090 / compute capability 12.0, build from source:

```bash
cd /mnt/c/Users/sully/Documents/projects/chatbot-uncensored
source .venv-wsl/bin/activate

pip uninstall -y llama-cpp-python

export CUDA_HOME=/usr/local/cuda-12.9
export CUDAToolkit_ROOT=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_NATIVE=OFF" \
FORCE_CMAKE=1 \
pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python
```

Check CUDA linking:

```bash
ldd .venv-wsl/lib/python3.12/site-packages/llama_cpp/lib/libllama.so | grep -i cuda
```

Expected output should include:

```text
libggml-cuda.so
libcudart.so.12
libcuda.so.1
```

## Running the Chatbot

Make sure your model exists:

```text
models/NemoMix-Unleashed-12B-Q4_K_M.gguf
```

Run:

```bash
source .venv-wsl/bin/activate
python chatbot.py
```

A GPU-enabled `Llama` setup should use:

```python
llm = Llama(
    model_path=modelPath,
    n_ctx=8192,
    n_gpu_layers=-1,
    n_threads=8,
    verbose=False
)
```

To verify GPU usage, temporarily set:

```python
verbose=True
```

Look for logs like:

```text
CUDA0 compute buffer size
CUDA : ARCHS = 1200
BLACKWELL_NATIVE_FP4 = 1
```

## SOUL.md Persona File

`SOUL.md` controls the chatbot personality, behavior, and roleplay style.

Example:

```md
# SOUL

You are a local chatbot with a consistent persona.

## Style
- Stay in character.
- Keep responses clear and direct.
- Follow the user's roleplay scenario.
- Do not break character unnecessarily.

## Behavior
- Be helpful.
- Be conversational.
- Avoid unnecessary clarification questions.
```

The chatbot loads `SOUL.md` as the system prompt.

## RAG Setup

RAG allows the chatbot to answer using local documents.

Install ChromaDB:

```bash
pip install chromadb
```

Put documents in:

```text
docs/
```

Supported file examples:

```text
.txt
.md
.py
.json
.csv
.html
.css
.js
```

You also need an embedding GGUF model, for example:

```text
models/embedding-model.gguf
```

Good embedding model choices include:

```text
nomic-embed-text-v1.5 GGUF
bge-small-en GGUF
bge-base-en GGUF
```

Update `ingestDocs.py` if your embedding model has a different name:

```python
embeddingModelPath = "./models/embedding-model.gguf"
```

Run ingestion:

```bash
python ingestDocs.py
```

This creates or updates:

```text
chromaDb/
```

Then run the chatbot:

```bash
python chatbot.py
```

## Startup Scripts

### Windows PowerShell Startup

Create `start-chatbot-windows.ps1`:

```powershell
wsl -d Ubuntu -- bash -lc "cd /mnt/c/Users/sully/Documents/projects/chatbot-uncensored && ./start-chatbot.sh"
```

Run it from PowerShell:

```powershell
.\start-chatbot-windows.ps1
```

### WSL Startup Script

Create `start-chatbot.sh`:

```bash
#!/usr/bin/env bash

cd /mnt/c/Users/sully/Documents/projects/chatbot-uncensored || exit 1

export CUDA_HOME=/usr/local/cuda-12.9
export CUDAToolkit_ROOT=/usr/local/cuda-12.9
export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

source .venv-wsl/bin/activate

python chatbot.py

exec bash
```

Fix Windows line endings if needed:

```bash
sed -i 's/\r$//' start-chatbot.sh
chmod +x start-chatbot.sh
```

## Git Ignore

Do not commit virtual environments, models, or vector databases.

Recommended `.gitignore`:

```gitignore
# Python / WSL virtual envs
.venv/
.venv-wsl/
venv/
__pycache__/
*.pyc

# Local model files
models/
*.gguf
*.bin
*.safetensors

# RAG/vector DB
chromaDb/
*.sqlite3
*.db

# Local env/config
.env

# OS/editor junk
.DS_Store
Thumbs.db
.vscode/
.idea/
```

If `.venv-wsl` was accidentally staged:

```bash
git reset
git rm -r --cached --ignore-unmatch .venv-wsl
git add .
```

## Troubleshooting

### `Model path does not exist`

The model path in your script does not match the actual file.

Check:

```bash
ls -lh models
```

Then update:

```python
modelPath = "./models/your-model-name.gguf"
```

### `libcudart.so.12: cannot open shared object file`

CUDA runtime libraries are not found.

Set:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH
```

If using pip-installed NVIDIA CUDA libraries, also include their paths.

### `Illegal instruction`

This can happen with incompatible prebuilt wheels or unsafe CPU-native builds.

For RTX 5090, rebuild from source using:

```bash
-DGGML_CUDA=on
-DCMAKE_CUDA_ARCHITECTURES=120
-DGGML_NATIVE=OFF
```

### `Using fallback chat format: llama-2`

The model does not have a detected chat template.

Try setting the chat format manually:

```python
chat_format="mistral-instruct"
```

or:

```python
chat_format="chatml"
```

### Git tries to add `.venv-wsl`

Make sure `.gitignore` includes:

```gitignore
.venv-wsl/
```

Then run:

```bash
git reset
git rm -r --cached --ignore-unmatch .venv-wsl
git add .
```

## Notes

* `SOUL.md` controls behavior and personality.
* `docs/` contains knowledge files for RAG.
* `chromaDb/` stores the vector database.
* `models/` stores local GGUF models.
* `CMAKE_ARGS` is only needed when building or rebuilding `llama-cpp-python`.
* CUDA path exports are needed when starting the chatbot.
* GGUF models should not be committed to Git.

## License

This project is for local experimentation and personal development. Check the licenses of any GGUF models used before redistribution.
