# serve.py
import os
import numpy as np
import pandas as pd
import faiss
from typing import List
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from sklearn.preprocessing import normalize

ART = "./artifacts"
CATALOG = os.path.join(ART, "catalog.parquet")
EMB = os.path.join(ART, "emb.npy")
IDX = os.path.join(ART, "faiss.index")
IDMAP = os.path.join(ART, "id_to_idx.txt")

app = FastAPI(title="Product Recommender", version="1.0")

# --- load artifacts on startup
catalog = pd.read_parquet(CATALOG)
emb = np.load(EMB).astype(np.float32)  # (N,d), already L2-normalized
index = faiss.read_index(IDX)
id_to_idx = {}
with open(IDMAP, "r", encoding="utf-8") as f:
    for line in f:
        k,v = line.strip().split("\t")
        id_to_idx[k] = int(v)
idx_to_id = {v:k for k,v in id_to_idx.items()}

class RecItem(BaseModel):
    product_id: str
    score: float
    title: str
    tags: str
    image_path: str

class RecResponse(BaseModel):
    query_product_id: str
    results: List[RecItem]

def topk_from_vector(vec: np.ndarray, k: int = 10):
    # vec is (d,), must be L2-normalized for IP ~ cosine
    vec = vec.reshape(1, -1).astype(np.float32)
    vec = normalize(vec, norm="l2", copy=False)
    scores, idxs = index.search(vec, k)
    return idxs[0].tolist(), scores[0].tolist()

@app.get("/recommend", response_model=RecResponse)
def recommend(product_id: str = Query(...), k: int = Query(10, ge=1, le=100)):
    if product_id not in id_to_idx:
        raise HTTPException(status_code=404, detail="product_id not found")

    q_idx = id_to_idx[product_id]
    q_vec = emb[q_idx]  # (d,)
    idxs, scores = topk_from_vector(q_vec, k+1)  # +1 to skip the item itself

    out = []
    for idx, sc in zip(idxs, scores):
        if idx == q_idx:
            continue
        row = catalog.iloc[idx]
        out.append(RecItem(
            product_id=idx_to_id[idx],
            score=float(sc),
            title=str(row["title"]),
            tags=str(row["tags"]),
            image_path=str(row["image_path"])
        ))
        if len(out) == k:
            break
    return RecResponse(query_product_id=product_id, results=out)

@app.get("/")
def root():
    return {"ok": True, "msg": "Use /recommend?product_id=...&k=10"}
