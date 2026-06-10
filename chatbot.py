from pathlib import Path
import chromadb
from llama_cpp import Llama

chatModelPath = "./models/NemoMix-Unleashed-12B-Q4_K_M.gguf"
embeddingModelPath = "./models/embedding-model.gguf"
soulPath = Path("./SOUL.md")

dbDir = "./chromaDb"
collectionName = "localDocs"


def loadSoul() -> str:
    if not soulPath.exists():
        return "You are a helpful local chatbot."

    return soulPath.read_text(encoding="utf-8", errors="ignore").strip()


def getEmbedding(embedder: Llama, text: str) -> list[float]:
    result = embedder.create_embedding(text)
    return result["data"][0]["embedding"]


def retrieveContext(embedder: Llama, collection, query: str, topK: int = 5) -> str:
    queryEmbedding = getEmbedding(embedder, query)

    results = collection.query(
        query_embeddings=[queryEmbedding],
        n_results=topK,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    contextBlocks = []

    for index, document in enumerate(documents):
        metadata = metadatas[index] if index < len(metadatas) else {}
        distance = distances[index] if index < len(distances) else None

        source = metadata.get("source", "unknown")
        chunkIndex = metadata.get("chunkIndex", "unknown")

        contextBlocks.append(
            f"[Source: {source}, chunk: {chunkIndex}, distance: {distance}]\n{document}"
        )

    return "\n\n---\n\n".join(contextBlocks)


def buildRagPrompt(userInput: str, retrievedContext: str) -> str:
    if not retrievedContext.strip():
        return userInput

    return f"""
Use the retrieved context below when it is relevant.
If the context does not contain the answer, say that the documents do not contain enough information and then answer from general knowledge if appropriate.

Retrieved context:
{retrievedContext}

User question:
{userInput}
""".strip()


def main() -> None:
    soulPrompt = loadSoul()

    print("Loading chat model...")

    llm = Llama(
        model_path=chatModelPath,
        n_ctx=8192,
        n_gpu_layers=-1,
        n_threads=8,
        chat_format="mistral-instruct",
        verbose=False
    )

    print("Loading embedding model...")

    embedder = Llama(
        model_path=embeddingModelPath,
        embedding=True,
        n_ctx=2048,
        n_gpu_layers=-1,
        verbose=False
    )

    client = chromadb.PersistentClient(path=dbDir)
    collection = client.get_or_create_collection(name=collectionName)

    messages = [
        {
            "role": "system",
            "content": soulPrompt
        }
    ]

    print("Local RAG chatbot started. Type 'exit' to quit.")

    while True:
        userInput = input("\nYou: ").strip()

        if userInput.lower() in ["exit", "quit"]:
            break

        retrievedContext = retrieveContext(
            embedder=embedder,
            collection=collection,
            query=userInput,
            topK=5
        )

        ragPrompt = buildRagPrompt(userInput, retrievedContext)

        messages.append({
            "role": "user",
            "content": ragPrompt
        })

        response = llm.create_chat_completion(
            messages=messages,
            temperature=0.75,
            top_p=0.95,
            repeat_penalty=1.1,
            max_tokens=700,
            stop=["</s>", "User:", "You:"]
        )

        assistantReply = response["choices"][0]["message"]["content"].strip()

        print(f"\nBot: {assistantReply}")

        messages.append({
            "role": "assistant",
            "content": assistantReply
        })


if __name__ == "__main__":
    main()