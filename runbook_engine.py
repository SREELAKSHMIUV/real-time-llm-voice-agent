import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

print("Initializing RAG system....")
model= SentenceTransformer("paraphrase-MiniLM-L3-v2")

with open("runbook.json", "r", encoding="utf-8") as f:
    runbook = json.load(f)

documents=[ item["issue"]+" "+item["solution"] for item in runbook]
print("Generating embeddings....")
embeddings=model.encode(documents)
embeddings=np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)
dimension=embeddings.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("RAG system ready")
def search_runbook(query,threshold=0.6):
    print("Inside search_runbook function")
    print("Query:",query)

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, k=1)
    score=scores[0][0]
    best_index=indices[0][0]
    print("Score:",scores[0][0])
    print("Threshold:",threshold)
    if score >= threshold:
          print("Match found",runbook[best_index]["issue"])
          return runbook[best_index]   
    else:
        print("Match not found")
        return None
    
