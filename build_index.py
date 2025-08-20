# build_index.py
import os
import argparse
import numpy as np
import pandas as pd
import faiss

from embed import load_products

def build_faiss_index(emb: np.ndarray, nlist: int = 256, m: int = 32) -> faiss.Index:
    """
    IVF-PQ index for scalable ANN:
      - nlist: number of Voronoi cells (coarse quantizer)
      - m: PQ segments (each 8 bits)
    """
    d = emb.shape[1]
    quantizer = faiss.IndexFlatIP(d)  # cosine/IP after L2-normalize
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
    # Train on a sample (or all if small)
    index.train(emb)
    index.add(emb)
    return index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img_dir", required=True, help="root path for image files")
    ap.add_argument("--out_dir", default="./artifacts")
    ap.add_argument("--nlist", type=int, default=256)
    ap.add_argument("--m", type=int, default=32)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading & encoding products...")
    df, fused, id_to_idx, txt_pipe = load_products(args.csv, args.img_dir)

    # save id map and catalog
    df.to_parquet(os.path.join(args.out_dir, "catalog.parquet"))
    np.save(os.path.join(args.out_dir, "emb.npy"), fused)
    # persist id_to_idx
    with open(os.path.join(args.out_dir, "id_to_idx.txt"), "w", encoding="utf-8") as f:
        for k,v in id_to_idx.items():
            f.write(f"{k}\t{v}\n")
    # persist TF-IDF pipeline
    import joblib
    joblib.dump(txt_pipe, os.path.join(args.out_dir, "tfidf.joblib"))

    print("Building FAISS index...")
    index = build_faiss_index(fused, nlist=args.nlist, m=args.m)
    faiss.write_index(index, os.path.join(args.out_dir, "faiss.index"))

    print(f"Done. Artifacts saved to: {args.out_dir}")

if __name__ == "__main__":
    main()
