import os
import re
import json
import argparse
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch

from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM


# ----------------------------
# Text utils
# ----------------------------
def clean_text(x: str) -> str:
    if x is None:
        return ""
    x = str(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x

def normalize_tokens(text: str) -> List[str]:
    text = clean_text(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]

def jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))


# ----------------------------
# Category predictor (load-only)
# ----------------------------
class CategoryPredictor:
    def __init__(self, model_dir: str, device: str):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        with open(os.path.join(model_dir, "label_encoder.json"), "r", encoding="utf-8") as f:
            self.classes = json.load(f)

    def predict_topk(self, text: str, k: int = 3) -> List[Tuple[str, float]]:
        enc = self.tok([text], truncation=True, max_length=256, padding=True, return_tensors="pt")
        enc = {kk: vv.to(self.device) for kk, vv in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().numpy()[0]
        idxs = np.argsort(-probs)[:k]
        return [(self.classes[i], float(probs[i])) for i in idxs]


# ----------------------------
# Optional RAG-style expander
# ----------------------------
class QuestionExpander:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.gen = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    def expand(self, question_text: str, snippets: List[str]) -> str:
        prompt = (
            "Generate a short answer-like expansion with useful keywords and paraphrases.\n\n"
            f"Question: {question_text}\n\n"
            f"Related snippets: {' '.join(snippets[:3])}\n\n"
            "Expansion:"
        )
        enc = self.tok(prompt, truncation=True, max_length=512, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.gen.generate(**enc, max_new_tokens=80, num_beams=4, do_sample=False)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()


# ----------------------------
# Hybrid Retriever
# ----------------------------
# class HybridRetriever:
#     def __init__(self, df: pd.DataFrame):
#         self.df = df.copy()
#         for c in ["subject", "body", "answer"]:
#             self.df[c] = self.df[c].fillna("").map(clean_text)
#         self.df["category"] = self.df["category"].astype(str)

#         self.df["doc_text"] = (self.df["subject"] + " " + self.df["body"] + " " + self.df["answer"]).str.strip()
#         tokenized = [normalize_tokens(t) for t in self.df["doc_text"].tolist()]
#         self.bm25 = BM25Okapi(tokenized)

#     def retrieve_snippets(self, query: str, k: int = 5) -> List[str]:
#         scores = self.bm25.get_scores(normalize_tokens(query))
#         top = np.argsort(-scores)[:k]
#         snippets = []
#         for i in top:
#             snippets.append(f"Q:{self.df.iloc[i]['subject']} A:{self.df.iloc[i]['answer'][:140]}")
#         return snippets

#     def rank(
#         self,
#         subject: str,
#         body: str,
#         predicted_cats: List[Tuple[str, float]],
#         trans_dict: Dict[str, Dict[str, Dict[str, float]]],
#         expander: Optional[QuestionExpander] = None,
#         candidate_k: int = 300,
#         final_k: int = 10,
#         subject_sim_threshold: float = 0.75,   # stricter than before
#         unique_answers: bool = True,           # NEW: remove duplicate answers
#     ) -> pd.DataFrame:

#         query = clean_text(subject + " " + body)
#         q_tokens = normalize_tokens(query)

#         # dictionary expansion from top category
#         top_cat = predicted_cats[0][0]
#         d = trans_dict.get(str(top_cat), {})
#         dict_tokens = []
#         for s in q_tokens:
#             if s in d:
#                 dict_tokens.extend([t for t, _ in sorted(d[s].items(), key=lambda x: x[1], reverse=True)[:3]])

#         # optional RAG-style generation
#         gen_exp = ""
#         if expander is not None:
#             gen_exp = expander.expand(query, self.retrieve_snippets(query, k=5))

#         expanded_query = query + " " + " ".join(dict_tokens) + " " + gen_exp
#         eq_tokens = normalize_tokens(expanded_query)

#         # BM25 candidates
#         base_scores = self.bm25.get_scores(eq_tokens)
#         cand_idx = np.argsort(-base_scores)[:candidate_k].tolist()

#         # final score = BM25 + category boost + subject overlap boost
#         cat_prob = {c: p for c, p in predicted_cats}
#         final_scores = np.array(base_scores, dtype=float)

#         for i in cand_idx:
#             row = self.df.iloc[i]
#             final_scores[i] += 0.50 * float(cat_prob.get(row["category"], 0.0))
#             overlap = len(set(q_tokens) & set(normalize_tokens(row["subject"])))
#             final_scores[i] += 0.05 * overlap

#         # Sort by final score
#         cand_idx = sorted(cand_idx, key=lambda i: final_scores[i], reverse=True)

#         # Select with de-duplication:
#         selected = []
#         seen_answer_keys = set()
#         kept_subject_tokens = []

#         for i in cand_idx:
#             if len(selected) >= final_k:
#                 break

#             row = self.df.iloc[i]

#             # 1) Answer de-duplication (same answer text => skip)
#             if unique_answers:
#                 ans_key = re.sub(r"\s+", " ", row["answer"].lower()).strip()
#                 if ans_key in seen_answer_keys:
#                     continue

#             # 2) Subject similarity de-duplication
#             stoks = normalize_tokens(row["subject"])
#             if any(jaccard(stoks, kt) >= subject_sim_threshold for kt in kept_subject_tokens):
#                 continue

#             selected.append(i)
#             kept_subject_tokens.append(stoks)
#             if unique_answers:
#                 seen_answer_keys.add(ans_key)

#         # Build output table
#         rows = []
#         for rank, i in enumerate(selected, start=1):
#             r = self.df.iloc[i]
#             rows.append({
#                 "rank": rank,
#                 "qid": r["qid"],
#                 "category": r["category"],
#                 "score": round(float(final_scores[i]), 4),
#                 "subject": r["subject"],
#                 "body": r["body"],
#                 "answer": r["answer"],
#             })
#         return pd.DataFrame(rows)

class HybridRetriever:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        for c in ["subject", "body", "answer", "keywords"]:
            if c in self.df.columns:
                self.df[c] = self.df[c].fillna("").map(clean_text)
            else:
                self.df[c] = ""

        self.df["category"] = self.df["category"].astype(str)

        # ---- Metadata-weighted document text ----
        # Repeat keywords to give them higher BM25 influence
        def build_doc_text(r):
            kw = r["keywords"]
            kw_weighted = (" " + kw) * 3 if kw else ""   # weight metadata x3
            return (r["subject"] + " " + r["body"] + " " + r["answer"] + kw_weighted).strip()

        self.df["doc_text"] = self.df.apply(build_doc_text, axis=1)
        tokenized = [normalize_tokens(t) for t in self.df["doc_text"].tolist()]
        self.bm25 = BM25Okapi(tokenized)

    def rank(
        self,
        subject: str,
        body: str,
        predicted_cats: List[Tuple[str, float]],
        trans_dict: Dict[str, Dict[str, Dict[str, float]]],
        expander: Optional[QuestionExpander] = None,
        candidate_k: int = 300,
        final_k: int = 10,
        subject_sim_threshold: float = 0.75,
        unique_answers: bool = True,
    ) -> pd.DataFrame:

        query = clean_text(subject + " " + body)
        q_tokens = normalize_tokens(query)

        # Extract possible metadata tokens from query (simple heuristic)
        query_kw_tokens = set(q_tokens)

        # dictionary expansion from top category
        top_cat = predicted_cats[0][0]
        d = trans_dict.get(str(top_cat), {})
        dict_tokens = []
        for s in q_tokens:
            if s in d:
                dict_tokens.extend([t for t, _ in sorted(d[s].items(), key=lambda x: x[1], reverse=True)[:3]])

        gen_exp = ""
        if expander is not None:
            gen_exp = expander.expand(query, [])

        expanded_query = query + " " + " ".join(dict_tokens) + " " + gen_exp
        eq_tokens = normalize_tokens(expanded_query)

        base_scores = self.bm25.get_scores(eq_tokens)
        cand_idx = np.argsort(-base_scores)[:candidate_k].tolist()

        cat_prob = {c: p for c, p in predicted_cats}
        final_scores = np.array(base_scores, dtype=float)

        # Dedup controls
        selected = []
        seen_answer_keys = set()
        kept_subject_tokens = []

        for i in cand_idx:
            if len(selected) >= final_k:
                break

            row = self.df.iloc[i]

            # Category boost
            final_scores[i] += 0.50 * float(cat_prob.get(row["category"], 0.0))

            # Subject overlap boost
            overlap = len(set(q_tokens) & set(normalize_tokens(row["subject"])))
            final_scores[i] += 0.05 * overlap

            # ✅ Metadata boost: overlap between query tokens and stored keywords
            kw_tokens = set(normalize_tokens(row["keywords"]))
            meta_overlap = len(query_kw_tokens & kw_tokens)
            final_scores[i] += 0.25 * meta_overlap   # strong metadata boost

            # Answer dedupe (optional)
            if unique_answers:
                ans_key = re.sub(r"\s+", " ", row["answer"].lower()).strip()
                if ans_key in seen_answer_keys:
                    continue

            # Subject similarity dedupe (Jaccard)
            stoks = normalize_tokens(row["subject"])
            if any(jaccard(stoks, kt) >= subject_sim_threshold for kt in kept_subject_tokens):
                continue

            selected.append(i)
            kept_subject_tokens.append(stoks)
            if unique_answers:
                seen_answer_keys.add(ans_key)

        rows = []
        for rank, i in enumerate(selected, start=1):
            r = self.df.iloc[i]
            rows.append({
                "rank": rank,
                "qid": r["qid"],
                "category": r["category"],
                "score": round(float(final_scores[i]), 4),
                "subject": r["subject"],
                "keywords": r["keywords"],
                "answer": r["answer"],
            })
        return pd.DataFrame(rows)

# ----------------------------
# Pretty printing (readable!)
# ----------------------------
# def print_results(pred: List[Tuple[str, float]], results: pd.DataFrame):
#     print("\n==============================")
#     print(" Predicted Categories (Top-3)")
#     print("==============================")
#     for c, p in pred:
#         print(f"- {c:11s} : {p:.3f}")

#     print("\n=============================================")
#     print(" Top Retrieved Results (Unique & Readable)")
#     print("=============================================")

#     if results.empty:
#         print("No results found (try increasing candidate_k).")
#         return

#     for _, row in results.iterrows():
#         print(f"\n[{int(row['rank'])}] QID: {row['qid']} | Category: {row['category']} | Score: {row['score']}")
#         print(f"Subject: {row['subject']}")
#         print(f"Body   : {row['body']}")
#         print(f"Answer : {row['answer']}")
#         print("-" * 60)

def print_results(pred, results: pd.DataFrame):
    print("\n==============================")
    print(" Predicted Categories (Top-3)")
    print("==============================")
    for c, p in pred:
        print(f"- {c:11s} : {p:.3f}")

    print("\n=============================================")
    print(" Top Retrieved Results (Unique & Readable)")
    print("=============================================")

    if results is None or results.empty:
        print("No results found.")
        return

    has_body = "body" in results.columns
    has_keywords = "keywords" in results.columns

    for _, row in results.iterrows():
        print(f"\n[{int(row.get('rank', 0))}] QID: {row.get('qid')} | Category: {row.get('category')} | Score: {row.get('score')}")
        print(f"Subject: {row.get('subject')}")

        if has_body:
            print(f"Body   : {row.get('body')}")
        if has_keywords:
            print(f"Meta(K): {row.get('keywords')}")

        print(f"Answer : {row.get('answer')}")
        print("-" * 60)



# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="../data/cqa.csv")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--use_expansion", action="store_true")
    ap.add_argument("--gen_model", type=str, default="google/flan-t5-base")
    ap.add_argument("--query_subject", type=str, default="WiFi keeps dropping")
    ap.add_argument("--query_body", type=str, default="My laptop loses wireless connection every 5 minutes.")
    ap.add_argument("--final_k", type=int, default=10)
    ap.add_argument("--candidate_k", type=int, default=300)
    ap.add_argument("--subject_sim_threshold", type=float, default=0.75)
    ap.add_argument("--allow_duplicate_answers", action="store_true")  # if you want to disable answer dedupe
    args = ap.parse_args()

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[Info] device={device}")

    df = pd.read_csv(args.data_csv)

    # cat = CategoryPredictor("../artifacts/category_model", device=device)
    # with open("../artifacts/translation_dict/cat_translation_dict.json", "r", encoding="utf-8") as f:
    #     trans = json.load(f)

    cat = CategoryPredictor("artifacts/category_model", device=device)
    with open("artifacts/translation_dict/cat_translation_dict.json", "r", encoding="utf-8") as f:
        trans = json.load(f)


    expander = QuestionExpander(args.gen_model, device=device) if args.use_expansion else None
    retriever = HybridRetriever(df)

    q_text = clean_text(args.query_subject + " [SEP] " + args.query_body)
    pred = cat.predict_topk(q_text, k=3)

    results = retriever.rank(
        args.query_subject,
        args.query_body,
        pred,
        trans,
        expander=expander,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
        subject_sim_threshold=args.subject_sim_threshold,
        unique_answers=(not args.allow_duplicate_answers),
    )

    print_results(pred, results)


if __name__ == "__main__":
    main()
