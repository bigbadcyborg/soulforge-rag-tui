# SoulForge TUI (Chatbot Uncensored)

A local-first, TUI-based chatbot running GGUF models through `llama-cpp-python`. Supports CUDA GPU acceleration, persona control via `SOUL.md`, and interactive, toggleable RAG with ChromaDB.

This project is designed to run locally (Windows/WSL) with CUDA acceleration for NVIDIA GPUs.

## Features

* **GGUF Inference**: Local `llama-cpp-python` backend for high-performance GGUF models.
* **Textual TUI**: A structured terminal user interface with chat history, status bar, and interactive commands.
* **Persona Control**: `SOUL.md` defines your assistant's identity, personality, and tone.
* **Runtime Feature Toggles**: Enable or disable RAG, memory, SOUL, streaming, and more via `/features` (auto-saves to `config.yaml`).
* **Toggleable RAG**: Enable/disable RAG at runtime with `/rag` or the feature menu.
* **Interactive Doc Selection**: Use `/rag` to select specific documents from your vector store via a checkbox-based modal.
* **Document Ingestion**: Index `docs/` into ChromaDB with `/ingest` or `python ingestDocs.py` (text + PDF/OCR).
* **Source Viewer**: Inspect retrieved chunks from the last question with `/sources`.
* **Tools Workshop**: Open `/tools` (or `/tool`) in TUI to add shell allowlist entries and run manual tool tests with visible output.
* **WSL 2 & CUDA support**: Optimized for NVIDIA GPUs (including Blackwell/RTX 5090) within WSL Ubuntu.
* **Persona Hot-Reload**: Update `SOUL.md` and reload the character instantly with `/reload-soul`.
* **Compute Indicator**: Status bar shows **GPU** or **CPU** after the model loads.

## Commands

Type `/help` in the TUI or CLI for the full list with descriptions. Key commands:

* `/ingest`: Index files from `docs/` into ChromaDB (text + PDF/OCR).
* `/sources`: View retrieved chunks from the last question.
* `/rag on` / `/rag off`: Enable or disable RAG retrieval.
* `/rag`: Toggle RAG or open document selection modal (TUI).
* `/rag all`: Enable RAG using all indexed documents.
* `/rag doc1.txt,doc2.txt`: Enable RAG filtered to specific documents.
* `/features`: Open feature toggle menu (TUI) or list flags (CLI).
* `/tools` or `/tool`: Open the tools workshop menu (TUI) or show tools status (CLI).
* `/tools test <name> '<json args>'`: Run a manual tool test (CLI).
* `/tools add-shell <command>`: Add a command prefix to `tools.shellAllowlist`.
* `/tools allowlist`: Show current shell allowlist entries.
* `/tools-log`: View recent tool execution audit log entries.
* `/status`: Show model, active features, and RAG index stats.
* `/reload-soul`: Refresh persona from `SOUL.md` without restarting.
* `/help`: Show all commands with descriptions.
* `/exit`: Quit the application.

## Project Structure

```text
chatbot-uncensored/
  app/                      # Core application package
    core/                   # Logic, config, and model runtime
    memory/                 # Persistent memory management
    rag/                    # ChromaDB retrieval logic
    tui/                    # Textual UI components and styles
  SOUL.md                   # Character persona and instructions
  config.yaml               # Model and feature configuration
  ingestDocs.py             # Script to index documents into ChromaDB
  start-chatbot.sh          # WSL startup script
  start-chatbot-windows.ps1 # Windows PowerShell startup script
  docs/                     # Source documents for RAG
  models/                   # GGUF model files
  chromaDb/                 # ChromaDB vector store
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

## Quick start (recommended)

These steps assume Windows 11 with WSL Ubuntu installed.

1. **Clone the repo (Windows or WSL)**  
   ```bash
   git clone https://github.com/bigbadcyborg/chatbot.git
   cd chatbot
   ```

2. **Run setup in WSL**  
   Inside Ubuntu on WSL:
   ```bash
   ./setup.sh
   ```
   This script wraps `install-wsl.sh`, installs apt packages, creates `.venv-wsl`, and installs Python dependencies from `requirements.txt`.  
   For a CUDA build of `llama-cpp-python`, run:
   ```bash
   ./setup.sh --with-cuda
   ```

3. **Run basic checks**  
   Still inside WSL:
   ```bash
   ./doctor.sh
   ```
   The doctor script checks common issues (missing venv/deps, config, models, CUDA, and Git hygiene) and prints remediation tips.

4. **Start the chatbot (Windows or WSL)**  
   - From **Windows PowerShell** (recommended once WSL is set up):
     ```powershell
     .\start-chatbot-windows.ps1
     ```
   - Or directly inside **WSL**:
     ```bash
     ./start-chatbot.sh
     ```

After startup, type `/help` in the TUI for all commands, or try `/status`, `/ingest`, `/rag on`, and `/tools`.

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
python -m app.main
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

Install Python dependencies (includes ChromaDB and PDF/OCR libraries):

```bash
pip install -r requirements.txt
```

For scanned PDF OCR, install system packages in WSL:

```bash
sudo apt install -y tesseract-ocr poppler-utils
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
.pdf
```

PDF support uses a two-stage pipeline: fast text extraction for digital PDFs, with automatic OCR fallback for scanned documents.

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

Run ingestion (from project root):

```bash
python ingestDocs.py
```

Or from inside the running TUI/CLI:

```text
/ingest
```

This creates or updates:

```text
chromaDb/
```

Then run the chatbot:

```bash
python -m app.main
```

Enable RAG and ask questions:

```text
/rag on
```

## Tool Harness / Tools Workshop

The optional tool harness lets the assistant propose or run controlled local tools.

Default behavior:

* `features.tools` is off by default.
* Risky tools still require approval.
* Shell tools are blocked unless both `allowShell: true` and `shellAllowlist` contains matching command prefixes.
* Tool events are logged to `logs/tool_calls.jsonl`.

Enable tools in `config.yaml`:

```yaml
features:
  tools: true

tools:
  allowWrite: false
  allowShell: true
  shellAllowlist:
    - git status
```

In TUI:

* Run `/tools` (or `/tool`) to open the workshop.
* Use **Add Shell Command** to append a command prefix.
* Use **Test Tool** to run a manual test and inspect output.

## Doctor script

After running `./setup.sh`, you can use the doctor script to validate your environment:

```bash
./doctor.sh
```

The script runs a series of read-only checks and reports **OK/WARN/FAIL** for:

* WSL environment (Linux, not Git Bash)
* Presence of `.venv-wsl` and key Python packages
* Optional CUDA support (`nvidia-smi`, `libllama.so` linkage)
* Existence and basic validity of `config.yaml`, model paths, and `docs/`
* Git hygiene (no `models/`, `chromaDb/`, `.venv-wsl/`, or `logs/` staged)

Use the output to fix issues, then re-run `./doctor.sh` until everything is green.

## Example starter files

Iteration 14 includes starter assets under `examples/`:

* `examples/config.example.yaml`
* `examples/SOUL.example.md`
* `examples/docs/example.txt`
* `examples/skills/example_skill.md`

Copy them into place:

```bash
cp examples/config.example.yaml config.yaml
cp examples/SOUL.example.md SOUL.md
mkdir -p docs
cp -r examples/docs/* docs/
```

Optional: copy the sample skill for testing:

```bash
mkdir -p app/skills/active
cp examples/skills/example_skill.md app/skills/active/example_skill.md
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

python -m app.main

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
