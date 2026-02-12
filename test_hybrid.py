
import narrative_logic as logic
import db_manager as db
import time

def test_hybrid_search():
    print("=== Testing Hybrid Search Implementation ===")
    
    # Check if dependencies are loaded
    print(f"FAISS available: {logic.HAS_FAISS}")
    print(f"Annoy available: {logic.HAS_Annoy if hasattr(logic, 'HAS_Annoy') else logic.HAS_ANNOY}")
    
    # 1. Test Embedding Cache
    print("\n[1] Testing Embedding Cache...")
    ids, X, X_norm = logic.load_embeddings_cache()
    print(f"Loaded {len(ids)} embeddings. Matrix shape: {X.shape}")
    
    if len(ids) == 0:
        print("Warning: No embeddings found. Please add some logs first.")
        return

    # 2. Test Index Building
    print("\n[2] Testing Index Building...")
    typ, idx, ids_map = logic.load_or_build_index()
    print(f"Index type loaded: {typ}")
    
    # 3. Test Hybrid Search Query
    query = "I feel anxious about my promise"
    print(f"\n[3] Running Hybrid Search for query: '{query}'")
    
    start_time = time.time()
    results = logic.find_related_logs(query, top_k=5)
    end_time = time.time()
    
    print(f"Search took {(end_time - start_time)*1000:.2f} ms")
    print(f"Found {len(results)} results.")
    
    for i, res in enumerate(results):
        print(f"  {i+1}. [{res.get('similarity', 0):.4f}] {res['content'][:50]}...")

    print("\n=== Test Complete ===")
    
if __name__ == "__main__":
    test_hybrid_search()
