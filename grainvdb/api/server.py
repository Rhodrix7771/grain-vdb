from sanic import Sanic, response
import torch
import numpy as np
import time
import orjson
from grain_vdb import GrainVDB

# GrainVDB HFT API Server
# ======================
# Sub-ms overhead for local vector lookups.

app = Sanic("GrainVDB_API")
vdb = None

# In-memory startup for speed
@app.before_server_start
async def setup_vdb(app, loop):
    global vdb
    DIM = 128
    vdb = GrainVDB(dim=DIM)
    
    # Pre-warm with demo data (100k vectors)
    print("Pre-warming GrainVDB with 100k vectors...")
    dummy_data = np.random.randn(100000, DIM).astype(np.float32)
    vdb.add_vectors(dummy_data)
    print("GrainVDB Warmup Complete.")

@app.post("/search")
async def search(request):
    """
    Search endpoint: Ingests vector, returns top-k and Sheaf consistency score.
    """
    try:
        data = request.json
        query_vec = np.array(data.get("vector")).astype(np.float32)
        k = int(data.get("k", 5))
        
        # Execute query on Metal
        scores, indices, elapsed = vdb.query(query_vec, k=k)
        
        # Sheaf Consistency Check (Standard for Premium tier)
        # Note: In production, this would use the real adjacency graph
        is_consistent = scores[0].item() > 0.8 # Placeholder logic
        
        return response.json({
            "status": "success",
            "results": [
                {"id": int(idx), "score": float(score)} 
                for score, idx in zip(scores, indices)
            ],
            "latency_ms": elapsed * 1000,
            "consistency": "HIGH" if is_consistent else "FRACTURED"
        }, status=200)
        
    except Exception as e:
        return response.json({"status": "error", "message": str(e)}, status=400)

@app.get("/health")
async def health(request):
    return response.json({
        "status": "healthy",
        "device": vdb.device,
        "vector_count": vdb.vectors.shape[0] if vdb.vectors is not None else 0
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, workers=1, access_log=False)
