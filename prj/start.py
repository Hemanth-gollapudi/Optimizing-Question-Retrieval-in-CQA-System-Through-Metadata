import numpy as np, faiss, torch, transformers, pandas, sklearn
from sentence_transformers import SentenceTransformer
print("numpy", np.__version__)
print("faiss", faiss.__version__)
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
print("transformers", transformers.__version__)

m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
emb = m.encode(["hello world", "how are you"], convert_to_numpy=True)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
D,I = index.search(emb, k=1)
print("FAISS ok:", D.shape, I.shape)