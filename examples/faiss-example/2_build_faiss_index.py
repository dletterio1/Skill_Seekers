#!/usr/bin/env python3
"""Build FAISS index with OpenAI embeddings"""
import json, sys, os
import numpy as np
from pathlib import Path

try:
    import faiss
    from openai import OpenAI
    from rich.console import Console
except ImportError:
    print("❌ Missing dependencies! Run: pip install -r requirements.txt")
    sys.exit(1)

console = Console()

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    console.print("[red]❌ OPENAI_API_KEY not set![/red]")
    console.print("Set it with: export OPENAI_API_KEY=sk-...")
    sys.exit(1)

# Load data
console.print("📥 Loading skill data...")
with open("output/flask-faiss.json") as f:
    data = json.load(f)

documents = data["documents"]
metadatas = data["metadatas"]
ids = data["ids"]

console.print(f"✅ Loaded {len(documents)} documents")

# Generate embeddings
console.print("\n🔄 Generating embeddings (this may take 30-60 seconds)...")
console.print(f"   Cost: ~$0.001 for {len(documents)} documents")

client = OpenAI(api_key=api_key)
embeddings = []

for i, doc in enumerate(documents):
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=doc[:8000]  # Truncate to max length
    )
    embeddings.append(response.data[0].embedding)

    if (i + 1) % 5 == 0:
        console.print(f"   Progress: {i+1}/{len(documents)}")

console.print("✅ Embeddings generated!")

# Build FAISS index
console.print("\n🏗️  Building FAISS index...")
dimension = len(embeddings[0])  # 1536 for ada-002
vectors = np.array(embeddings).astype('float32')

# Create index (L2 distance)
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# Save everything
faiss.write_index(index, "flask.index")
with open("flask_metadata.json", "w") as f:
    json.dump({"documents": documents, "metadatas": metadatas, "ids": ids}, f)

console.print(f"✅ Index saved: flask.index")
console.print(f"✅ Metadata saved: flask_metadata.json")
console.print(f"\n💡 Total vectors: {index.ntotal}")
console.print(f"💡 Dimension: {dimension}")
console.print("\n➡️  Next: python 3_query_example.py")
