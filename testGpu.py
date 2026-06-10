from llama_cpp import Llama

modelPath = "./models/NemoMix-Unleashed-12B-Q4_K_M.gguf"

llm = Llama(
    model_path=modelPath,
    n_ctx=4096,
    n_gpu_layers=-1,
    verbose=True
)

print("Loaded")