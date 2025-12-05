# src/evaluate.py
import argparse
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer


def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_answer(s: str) -> str:
    s = clean_text(str(s)).lower()
    s = re.sub(r"[\W_]+", "", s)
    return s


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["qid", "id", "question_id"]:
            ren[c] = "qid"
        elif cl in ["category", "label", "class"]:
            ren[c] = "category"
        elif cl in ["subject", "title"]:
            ren[c] = "subject"
        elif cl in ["question", "body", "query_body", "text"]:
            ren[c] = "question"
        elif cl in ["answer", "response", "completion"]:
            ren[c] = "answer"
    return df.rename(columns=ren)


class SemanticRetriever:
    def __init__(self, df: pd.DataFrame, embed_model: str):
        self.df = df.copy()

        if "answer" not in self.df.columns:
            raise ValueError("CSV must contain an 'answer' column.")
        if "subject" not in self.df.columns and "question" not in self.df.columns:
            raise ValueError("CSV must contain at least 'subject' or 'question' column.")

        sub = self.df["subject"].fillna("").astype(str) if "subject" in self.df.columns else ""
        body = self.df["question"].fillna("").astype(str) if "question" in self.df.columns else ""
        self.df["_retrieval_text"] = (sub + " " + body).map(clean_text)

        self.embedder = SentenceTransformer(embed_model)
        emb = self.embedder.encode(self.df["_retrieval_text"].tolist(), batch_size=64, show_progress_bar=True)
        self.emb = normalize(np.asarray(emb), norm="l2")

    def scores(self, query: str) -> np.ndarray:
        q_emb = self.embedder.encode([query], show_progress_bar=False)
        q_emb = normalize(np.asarray(q_emb), norm="l2")
        return (self.emb @ q_emb.T).reshape(-1)


def average_precision_at_k(ranked_indices, relevant_set, k: int) -> float:
    """
    AP@K for possibly-multiple relevant docs.
    AP = average of precision@rank over ranks where a relevant doc occurs.
    We limit to top K ranks.
    """
    hits = 0
    precisions = []
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        if idx in relevant_set:
            hits += 1
            precisions.append(hits / rank)
    if not precisions:
        return 0.0
    return float(np.mean(precisions))


def recall_at_k(ranked_indices, relevant_set, k: int) -> float:
    """Recall@K = (# relevant retrieved in top K) / (# relevant total)."""
    if not relevant_set:
        return 0.0
    got = sum(1 for idx in ranked_indices[:k] if idx in relevant_set)
    return got / len(relevant_set)


def build_relevant_sets(df: pd.DataFrame, relevance: str):
    """
    Returns a list where rel[i] is a set of indices relevant to query i.
    default relevance='answer' means: all rows with same normalized answer.
    """
    n = len(df)
    rel = [set() for _ in range(n)]

    if relevance == "qid":
        if "qid" not in df.columns:
            raise ValueError("relevance='qid' requires a qid column")
        groups = {}
        for i, qid in enumerate(df["qid"].tolist()):
            groups.setdefault(qid, []).append(i)
        for _, idxs in groups.items():
            s = set(idxs)
            for i in idxs:
                rel[i] = set(s)

    elif relevance == "category":
        if "category" not in df.columns:
            raise ValueError("relevance='category' requires a category column")
        groups = {}
        for i, cat in enumerate(df["category"].fillna("").astype(str).tolist()):
            groups.setdefault(cat, []).append(i)
        for _, idxs in groups.items():
            s = set(idxs)
            for i in idxs:
                rel[i] = set(s)

    else:  # answer
        ans_norm = df["answer"].map(norm_answer).tolist()
        groups = {}
        for i, a in enumerate(ans_norm):
            groups.setdefault(a, []).append(i)
        for _, idxs in groups.items():
            s = set(idxs)
            for i in idxs:
                rel[i] = set(s)

    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--ood_threshold", type=float, default=0.55)

    # both styles supported
    ap.add_argument("--ks", default=None, help="Comma-separated Ks, e.g. 1,2,5,10")
    ap.add_argument("--limit", type=int, default=0)

    ap.add_argument("--k", type=int, default=None)  # old style
    ap.add_argument("--num_queries", type=int, default=None)  # old style

    # realism controls
    ap.add_argument("--exclude_self", action="store_true", help="Exclude the query row from candidates (recommended)")
    ap.add_argument("--relevance", choices=["answer", "category", "qid"], default="answer",
                    help="How to define relevant docs for each query")
    args = ap.parse_args()

    # K values
    if args.k is not None:
        ks = (int(args.k),)
    elif args.ks is not None:
        ks = tuple(int(x.strip()) for x in args.ks.split(",") if x.strip())
    else:
        ks = (1, 2, 5, 10)

    # limit
    limit = args.num_queries if args.num_queries is not None else args.limit

    df = pd.read_csv(args.data_csv)
    df = normalize_columns(df)
    if limit and limit > 0:
        df = df.head(int(limit)).copy()

    retriever = SemanticRetriever(df, embed_model=args.embed_model)
    relevant_sets = build_relevant_sets(df, relevance=args.relevance)

    n = len(df)
    max_k = max(ks)

    map_sum = {k: 0.0 for k in ks}
    recall_sum = {k: 0.0 for k in ks}

    ood_count = 0
    avg_rel = 0.0

    for i, row in df.iterrows():
        subject = clean_text(row.get("subject", ""))
        question = clean_text(row.get("question", ""))
        query = clean_text(subject + " " + question)

        scores = retriever.scores(query)
        # OOD estimate from best *other* candidate (if excluding self)
        if args.exclude_self and i < len(scores):
            scores2 = scores.copy()
            scores2[i] = -1e9
            best = float(np.max(scores2))
        else:
            best = float(np.max(scores))

        if best < args.ood_threshold:
            ood_count += 1

        ranked = np.argsort(-scores)

        if args.exclude_self:
            ranked = [idx for idx in ranked if idx != i]

        rel = set(relevant_sets[i])
        if args.exclude_self:
            rel.discard(i)

        avg_rel += len(rel)

        for k in ks:
            recall_sum[k] += recall_at_k(ranked, rel, k)
            map_sum[k] += average_precision_at_k(ranked, rel, k)

    avg_rel /= max(1, n)
    ood_rate = ood_count / max(1, n)

    print("\n====================")
    print(" Realistic Retrieval Evaluation")
    print("====================")
    print(f"Rows evaluated: {n}")
    print(f"Embedding model: {args.embed_model}")
    print(f"Relevance: {args.relevance}")
    print(f"Exclude self: {args.exclude_self}")
    print(f"Avg #relevant docs/query: {avg_rel:.2f}")
    print(f"OOD threshold: {args.ood_threshold:.2f}")
    print(f"OOD rate: {ood_rate:.3f}")

    print("\nRecall@K (avg over queries)")
    for k in ks:
        print(f"- Recall@{k}: {(recall_sum[k]/n):.4f}")

    print("\nMAP@K (avg over queries)")
    for k in ks:
        print(f"- MAP@{k}: {(map_sum[k]/n):.4f}")


if __name__ == "__main__":
    main()
