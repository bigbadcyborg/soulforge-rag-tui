from pathlib import Path
from llama_cpp import Llama

modelPath = "./models/NemoMix-Unleashed-12B-Q4_K_M.gguf"
soulPath = "./SOUL.md"


def loadSoul(filePath: str) -> str:
    path = Path(filePath)

    if not path.exists():
        raise FileNotFoundError(f"Could not find SOUL.md at: {filePath}")

    soulText = path.read_text(encoding="utf-8").strip()

    if not soulText:
        raise ValueError("SOUL.md is empty.")

    return soulText


soulPrompt = loadSoul(soulPath)

llm = Llama(
    model_path=modelPath,
    n_ctx=8192,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False
)

messages = [
    {
        "role": "system",
        "content": soulPrompt
    }
]

print("Local chatbot started. Type 'exit' to quit.")
print("Loaded SOUL.md successfully.")

while True:
    userInput = input("\nYou: ").strip()

    if userInput.lower() in ["exit", "quit"]:
        break

    messages.append({
        "role": "user",
        "content": userInput
    })

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.85,
        top_p=0.95,
        repeat_penalty=1.12,
        max_tokens=500,
        stop=[
            "</s>",
            "<|end_of_text|>",
            "<|eot_id|>",
            "User:",
            "You:"
        ]
    )

    assistantReply = response["choices"][0]["message"]["content"].strip()

    print(f"\nBot: {assistantReply}")

    messages.append({
        "role": "assistant",
        "content": assistantReply
    })