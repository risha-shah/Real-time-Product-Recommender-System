# embed.py
from __future__ import annotations
import os, io
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ImageEncoder(nn.Module):
    """
    ResNet18 backbone -> 512D embedding (frozen weights).
    """
    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for p in m.parameters():
            p.requires_grad = False
        # take everything except final fc
        self.backbone = nn.Sequential(*list(m.children())[:-1])  # -> (B, 512, 1, 1)
        self.out_dim = 512

        self.preproc = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=models.ResNet18_Weights.DEFAULT.transforms.mean,
                        std=models.ResNet18_Weights.DEFAULT.transforms.std),
        ])

    @torch.inference_mode()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(imgs).flatten(1)  # (B,512)
        feats = nn.functional.normalize(feats, dim=1)
        return feats

    @torch.inference_mode()
    def encode_paths(self, paths):
        batch = []
        embs = []
        for p in paths:
            try:
                img = Image.open(p).convert("RGB")
            except Exception:
                # fallback: blank image if missing/corrupt
                img = Image.new("RGB", (224,224), color=(255,255,255))
            batch.append(self.preproc(img))
            if len(batch) == 32:
                embs.append(self.forward(torch.stack(batch).to(DEVICE)).cpu())
                batch = []
        if batch:
            embs.append(self.forward(torch.stack(batch).to(DEVICE)).cpu())
        return torch.cat(embs, dim=0).numpy()  # (N,512)

def build_text_pipeline():
    """
    TF-IDF on (title + tags) -> StandardScaler -> L2 normalize
    Returns a sklearn Pipeline that outputs a dense float32 vector.
    """
    tfidf = TfidfVectorizer(
        max_features=4096,
        ngram_range=(1,2),
        min_df=2,
        token_pattern=r"(?u)\b\w+\b"
    )
    scaler = StandardScaler(with_mean=False)  # keep sparse
    pipe = Pipeline([("tfidf", tfidf), ("scaler", scaler)])
    return pipe

def fuse_embeddings(img_emb: np.ndarray, txt_emb: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    Weighted L2-normalized fusion: z = normalize( alpha*img + (1-alpha)*txt )
    Handles different dims by projecting text to image dim if needed.
    """
    # Make text dense
    if not isinstance(txt_emb, np.ndarray):
        txt_emb = txt_emb.toarray()
    # Dim align (project larger to smaller dim via PCA-ish random proj)
    d_img, d_txt = img_emb.shape[1], txt_emb.shape[1]
    if d_txt != d_img:
        # fast random projection to the smaller dimension
        d = min(d_img, d_txt)
        rng = np.random.default_rng(42)
        if d_txt > d:
            P = rng.standard_normal((d_txt, d)).astype(np.float32) / np.sqrt(d_txt)
            txt_proj = txt_emb @ P
            img_proj = img_emb[:, :d]
        else:
            P = rng.standard_normal((d_img, d)).astype(np.float32) / np.sqrt(d_img)
            img_proj = img_emb @ P
            txt_proj = txt_emb[:, :d]
    else:
        img_proj, txt_proj = img_emb, txt_emb

    # Normalize
    def l2n(x):
        n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
        return x / n

    img_n = l2n(img_proj)
    txt_n = l2n(txt_proj)
    z = alpha * img_n + (1.0 - alpha) * txt_n
    return l2n(z).astype(np.float32)

def load_products(csv_path: str, img_root: str) -> Tuple[pd.DataFrame, np.ndarray, Dict[str,int]]:
    """
    Returns df with resolved image paths + fused embeddings + id_to_idx map.
    """
    df = pd.read_csv(csv_path)
    assert {"product_id","image_path","title","tags"}.issubset(df.columns)
    df["abs_image"] = df["image_path"].apply(lambda p: p if os.path.isabs(p) else os.path.join(img_root, p))

    # Build image embeddings
    img_enc = ImageEncoder().to(DEVICE).eval()
    img_emb = img_enc.encode_paths(df["abs_image"].tolist())  # (N,512)

    # Build text pipeline on combined text
    texts = (df["title"].fillna("") + " " + df["tags"].fillna("")).tolist()
    txt_pipe = build_text_pipeline()
    txt_emb_sparse = txt_pipe.fit_transform(texts)  # (N,D)

    # Fuse
    fused = fuse_embeddings(img_emb=img_emb, txt_emb=txt_emb_sparse, alpha=0.6)  # (N,d)

    id_to_idx = {pid: i for i, pid in enumerate(df["product_id"].tolist())}
    return df, fused, id_to_idx, txt_pipe
