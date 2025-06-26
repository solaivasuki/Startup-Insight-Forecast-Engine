import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os

model = SentenceTransformer('all-MiniLM-L6-v2')

base_dir = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(base_dir, "data", "startups_2025.csv")
df = pd.read_csv(csv_path)

texts = df.apply(lambda row: f"{row['Startup Name']} in {row['Industry']} at {row['Headquarters']} with Rs.{row['Total Revenue']} Cr revenue.", axis=1)
embeddings = model.encode(texts.tolist())
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Ensure rag directory exists
rag_dir = os.path.join(base_dir, "rag")
os.makedirs(rag_dir, exist_ok=True)

# ✅ Save files in rag/
faiss.write_index(index, os.path.join(rag_dir, "startup_index.faiss"))
with open(os.path.join(rag_dir, "startup_texts.pkl"), "wb") as f:
    pickle.dump(texts.tolist(), f)
