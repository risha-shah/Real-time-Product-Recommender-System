Built a real-time product recommender that fuses image and text signals (ResNet18 + TF-IDF) with FAISS 
ANN search, served via FastAPI for top-N results at scale


Indexing: IVF+PQ compresses vectors and searches only in relevant clusters → fast queries on 100k–10M items.

Batching: ImageEncoder.encode_paths processes images in batches (32 by default).

Cosine similarity: normalize vectors → use FAISS inner product for speed.

GPU: If available, switch to faiss-gpu and move the index with faiss.index_cpu_to_gpu.
