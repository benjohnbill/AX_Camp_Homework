
import os
import json
import numpy as np
import sys

# Ensure parent directory is in path to import db_manager
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import db_manager as db
except ImportError:
    print("Error: Could not import db_manager. Run this script from the project root or tools/ directory.")
    sys.exit(1)

# Optional dependencies
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    print("Warning: faiss-cpu not installed. Skipping FAISS index.")

try:
    from annoy import AnnoyIndex
    HAS_ANNOY = True
except ImportError:
    HAS_ANNOY = False
    print("Warning: annoy not installed. Skipping Annoy index.")

DIM = 1536
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def build_indices():
    print("Loading embeddings from database...")
    ids, embeddings = db.get_all_embeddings_as_float32()
    
    if len(ids) == 0:
        print("No embeddings found in database.")
        return

    print(f"Loaded {len(ids)} embeddings. Shape: {embeddings.shape}")
    
    # Save IDs mapping
    ids_path = os.path.join(DATA_DIR, "index_ids.json")
    with open(ids_path, "w") as f:
        json.dump(ids, f)
    print(f"Saved ID mapping to {ids_path}")

    # Build FAISS Index
    if HAS_FAISS:
        print("Building FAISS index...")
        # L2 Normalize for Cosine Similarity using Inner Product (IP) index
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(DIM)
        index.add(embeddings)
        
        index_path = os.path.join(DATA_DIR, "index.faiss")
        faiss.write_index(index, index_path)
        print(f"Saved FAISS index to {index_path}")

    # Build Annoy Index
    if HAS_ANNOY:
        print("Building Annoy index...")
        # Annoy uses angular distance for cosine similarity
        t = AnnoyIndex(DIM, 'angular')
        for i, vector in enumerate(embeddings):
            t.add_item(i, vector)
        
        t.build(10) # 10 trees
        index_path = os.path.join(DATA_DIR, "index.ann")
        t.save(index_path)
        print(f"Saved Annoy index to {index_path}")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    build_indices()
