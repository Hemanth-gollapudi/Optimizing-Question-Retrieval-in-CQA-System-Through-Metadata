import argparse
import re
import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# =========================================================
# Utilities
# =========================================================

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def answer_key(s: str) -> str:
    return re.sub(r"[\W_]+", "", clean_text(s).lower())


def is_code_question(text: str) -> bool:
    keywords = [
        "code", "program", "write", "implement",
        "algorithm", "function", "class",
        "java", "python", "c++", "javascript", "js"
    ]
    t = text.lower()
    return any(k in t for k in keywords)


def strip_code_fences(text: str) -> str:
    text = re.sub(r"```[a-zA-Z]*", "", text)
    return text.strip("`\n ")


# =========================================================
# Ollama Client
# =========================================================

def ollama_generate(model: str, prompt: str, max_tokens: int = 512) -> str:
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        },
        timeout=300
    )
    r.raise_for_status()
    return r.json()["response"].strip()


# =========================================================
# Semantic Retriever (dataset-first)
# =========================================================

class SemanticRetriever:
    def __init__(self, df: pd.DataFrame, embed_model: str):
        self.df = df.copy()

        text = (
            self.df.get("subject", "").fillna("").astype(str)
            + " "
            + self.df.get("question", "").fillna("").astype(str)
        ).map(clean_text)

        self.embedder = SentenceTransformer(embed_model)
        emb = self.embedder.encode(text.tolist(), batch_size=64, show_progress_bar=True)
        self.emb = normalize(np.asarray(emb), norm="l2")

    def search(self, query: str, k: int):
        q_emb = self.embedder.encode([query], show_progress_bar=False)
        q_emb = normalize(np.asarray(q_emb), norm="l2")
        scores = (self.emb @ q_emb.T).reshape(-1)

        idx = np.argsort(-scores)[:k]
        out = self.df.iloc[idx].copy()
        out["score"] = scores[idx]
        return out


# =========================================================
# Main
# =========================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--query_subject", default="")
    ap.add_argument("--query_body", default="")
    ap.add_argument("--candidate_k", type=int, default=30)
    ap.add_argument("--final_k", type=int, default=2)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--ood_threshold", type=float, default=0.55)
    args = ap.parse_args()

    # ---------------- Load dataset ----------------
    df = pd.read_csv(args.data_csv)

    # Normalize column names
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["subject", "title"]:
            ren[c] = "subject"
        elif cl in ["question", "body", "text"]:
            ren[c] = "question"
        elif cl in ["answer", "response", "completion"]:
            ren[c] = "answer"
    df = df.rename(columns=ren)

    retriever = SemanticRetriever(df, args.embed_model)

    query = clean_text(args.query_subject + " " + args.query_body)

    # ---------------- Dataset search ----------------
    results = retriever.search(query, args.candidate_k)
    best = float(results.iloc[0]["score"]) if not results.empty else -1.0

    # ---------------- Dataset HIT ----------------
    if best >= args.ood_threshold:
        seen = set()
        shown = 0
        for _, row in results.iterrows():
            ans = str(row.get("answer", "")).strip()
            key = answer_key(ans)
            if key in seen:
                continue
            seen.add(key)

            print("Answer:")
            print(ans)
            shown += 1
            if shown >= args.final_k:
                break
        return

    # ---------------- Fallback to LLM ----------------
    print("\n=============================================")
    print(" Out-of-dataset question detected")
    print("=============================================\n")
    print(f"[Debug] best_similarity={best:.4f}")

    # ---------------- Code ----------------
    if is_code_question(query):
        prompt = (
            "Write COMPLETE, VALID, COMPILABLE code.\n"
            "No explanation. No markdown.\n"
            "Stop after code ends.\n\n"
            f"{query}\n"
        )
        out = ollama_generate("deepseek-coder:6.7b", prompt, max_tokens=700)
        print("Answer:\n" + strip_code_fences(out))
        return

    # ---------------- General Knowledge ----------------
    prompt = (
        "Give ONLY the final factual answer.\n"
        "One short sentence.\n"
        "No explanation.\n\n"
        f"Question: {query}\n"
        "Answer:"
    )
    out = ollama_generate("llama3:8b", prompt, max_tokens=80)
    print("Answer:\n" + out)


if __name__ == "__main__":
    main()
