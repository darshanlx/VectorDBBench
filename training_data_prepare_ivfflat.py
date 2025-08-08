import faiss
import numpy as np
from datasets import load_dataset
import itertools
import json

d = 768                          # Cohere embedding dimension
nlist = 1024                     # Number of clusters

train_size = 100000              # Number of vectors for training
index_id = "cohere_wiki_ivfflat"   # Unique ID for MySQL

# Load Cohere embeddings in streaming mode
streamed_ds = load_dataset(
    "Cohere/wikipedia-22-12-en-embeddings",
    split="train",
    streaming=True
)

# Load training vectors (first 100K)
print(f"Loading {train_size} vectors for training...")
train_batch = list(itertools.islice(streamed_ds, train_size))
train_vectors = np.stack([rec['emb'] for rec in train_batch], axis=0).astype('float32')

# Normalize vectors for cosine similarity
faiss.normalize_L2(train_vectors)

quantizer = faiss.IndexFlatIP(d)                   # L2 distance metric
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

print("Training Started")

# Train on subset (100K normalized vectors)
index.train(train_vectors[:100000])

# Extract index data for MySQL
print("Extracting index data...")


centroids = index.quantizer.reconstruct_n(0, nlist)

print(f"Coarse centroids shape: {centroids.shape}")

metadata_sql = f"""
INSERT INTO VECTORDB_DATA VALUES (
  '{index_id}', 'metadata', 0,
  JSON_OBJECT('version', 1, 'nlist', {nlist})
);
"""

quantizer_sqls = [
    f"INSERT INTO VECTORDB_DATA VALUES ("
    f"'{index_id}', 'quantizer', {i}, '{centroids[i].tolist()}'"
    f");"
    for i in range(nlist)
]

# Combine all SQL
full_sql = (
    metadata_sql + "\n" + 
    "\n".join(quantizer_sqls)
)

# Save to file
with open("cohere_wiki_ivfflat.sql", "w") as f:
    f.write(full_sql)

print(f"SQL for auxiliary table saved to cohere_wiki_ivfflat.sql")
print(f"Total centroids: {nlist}")