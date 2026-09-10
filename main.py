# main.py
import re
import chromadb
from chromadb.utils import embedding_functions
import ollama
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()
class QuestionRequest(BaseModel):
    question: str
client = chromadb.PersistentClient(path="db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="paraphrase-multilingual-mpnet-base-v2")
collection = client.get_or_create_collection("roadmap", embedding_function=sentence_transformer_ef)

def split_into_chunks(text: str) -> list[str]:
    # Делим текст по заголовкам "## " — каждый раздел станет отдельным куском
    chunks = re.split(r"^#{2,} ", text, flags=re.MULTILINE)
    # Убираем пустые куски (например, если текст начинается с "## ")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    return chunks

def ask_bot(question: str) -> str:
    result = collection.query(query_texts=[question], n_results=3)
    context = "\n---\n".join([doc for doc in result['documents'][0]])
    response = ollama.chat(
        model="qwen2.5:3b",
        messages=[{"role": "user", "content": f"Вот контекст:\n{context}\n\nОтветь на вопрос: {question}"}],
    )
    return response['message']['content']

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer = ask_bot(request.question)
    return {"answer": answer}

if __name__ == "__main__":
    with open("roadmap.md", "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = split_into_chunks(text)
    collection.add(
        documents=chunks,
        metadatas=[{"source": "roadmap.md"} for _ in chunks],
        ids=[f"chunk_{i}" for i in range(len(chunks))])