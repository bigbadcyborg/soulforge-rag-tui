from pathlib import Path
import hashlib
import chromadb
from llama_cpp import Llama

docsDir = Path("./docs")
dbDir = "./chromaDb"
collectionName = "localDocs"

embeddingModelPath = "./models/embedding-model.gguf"


def readTextFiles(folderPath: Path) -> list[tuple[str, str]]:
    supportedExtensions = {".txt", ".md", ".py", ".json", ".csv", ".html", ".css", ".js"}

    files = []
    for filePath in folderPath.rglob("*"):
        if filePath.is_file() and filePath.suffix.lower() in supportedExtensions:
            text = filePath.read_text(encoding="utf-8", errors="ignore")
            files.append((str(filePath), text))

    return files


def chunkText(text: str, chunkSize: int = 1200, overlap: int = 200) -> list[str]:
    chunks = []
    startIndex = 0

    while startIndex < len(text):
        endIndex = startIndex + chunkSize
        chunk = text[startIndex:endIndex].strip()

        if chunk:
            chunks.append(chunk)

        startIndex += chunkSize - overlap

    return chunks


def makeId(sourcePath: str, chunkIndex: int, chunk: str) -> str:
    rawValue = f"{sourcePath}:{chunkIndex}:{chunk}"
    return hashlib.sha256(rawValue.encode("utf-8")).hexdigest()


def getEmbedding(llm: Llama, text: str) -> list[float]:
    result = llm.create_embedding(text)
    return result["data"][0]["embedding"]


def main() -> None:
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

    files = readTextFiles(docsDir)

    if not files:
        print("No documents found in ./docs")
        return

    print(f"Found {len(files)} files.")

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for sourcePath, text in files:
        chunks = chunkText(text)

        for chunkIndex, chunk in enumerate(chunks):
            chunkId = makeId(sourcePath, chunkIndex, chunk)
            embedding = getEmbedding(embedder, chunk)

            ids.append(chunkId)
            documents.append(chunk)
            embeddings.append(embedding)
            metadatas.append({
                "source": sourcePath,
                "chunkIndex": chunkIndex
            })

    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(f"Indexed {len(ids)} chunks into ChromaDB.")


if __name__ == "__main__":
    main()