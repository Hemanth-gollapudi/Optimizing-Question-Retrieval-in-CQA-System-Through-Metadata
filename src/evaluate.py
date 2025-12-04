import argparse
import json
import os
import numpy as np
import pandas as pd
import torch

# We import the same classes/utilities you already use in demo.py
from demo import CategoryPredictor, HybridRetriever, QuestionExpander, clean_text


def average_precision_at_k(rels, k):
    rels = rels[:k]
    hit = 0
    s = 0.0
    for i, r in enumerate(rels, start=1):
        if r == 1:
            hit += 1
            s += hit / i
    return 0.0 if hit == 0 else s / hit


def reciprocal_rank_at_k(rels, k):
    rels = rels[:k]
    for i, r in enumerate(rels, start=1):
        if r == 1:
            return 1.0 / i
    return 0.0


def r_precision(rels, R):
    if R == 0:
        return 0.0
    topR = rels[:R]
    return float(sum(topR)) / float(R)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="data/cqa.csv")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--num_queries", type=int, default=200)
    ap.add_argument("--use_expansion", action="store_true")
    ap.add_argument("--gen_model", type=str, default="google/flan-t5-base")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.data_csv)
    if "topic" not in df.columns:
        raise ValueError("CSV must contain 'topic' column. Regenerate data using updated make_data.py.")

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"[Info] device={device}")

    cat_model_dir = os.path.join(args.out_dir, "category_model")
    dict_path = os.path.join(args.out_dir, "translation_dict", "cat_translation_dict.json")

    cat = CategoryPredictor(cat_model_dir, device=device)
    with open(dict_path, "r", encoding="utf-8") as f:
        trans = json.load(f)

    expander = QuestionExpander(args.gen_model, device=device) if args.use_expansion else None
    retriever = HybridRetriever(df)

    # Ground truth: relevant docs share same topic
    topic_to_qids = {}
    for _, row in df.iterrows():
        topic_to_qids.setdefault(str(row["topic"]), set()).add(str(row["qid"]))

    # Choose queries
    idxs = rng.choice(len(df), size=min(args.num_queries, len(df)), replace=False)

    ap_scores = []
    rr_scores = []
    rp_scores = []

    for idx in idxs:
        q = df.iloc[idx]
        qid = str(q["qid"])
        topic = str(q["topic"])

        relevant = set(topic_to_qids.get(topic, set()))
        relevant.discard(qid)  # exclude itself
        R = len(relevant)

        # Predict category
        q_text = clean_text(q["subject"] + " [SEP] " + q["body"])
        pred = cat.predict_topk(q_text, k=3)

        # Retrieve ranked list
        ranked = retriever.rank(
            subject=q["subject"],
            body=q["body"],
            predicted_cats=pred,
            trans_dict=trans,
            expander=expander,
            candidate_k=300,
            final_k=max(args.k, 50),     # get more than k for R-Prec
            subject_sim_threshold=0.75,
            unique_answers=False,        # IMPORTANT: disable answer-dedupe for fair IR evaluation
        )

        retrieved_qids = ranked["qid"].astype(str).tolist()
        rels = [1 if rid in relevant else 0 for rid in retrieved_qids]

        ap_scores.append(average_precision_at_k(rels, args.k))
        rr_scores.append(reciprocal_rank_at_k(rels, args.k))
        rp_scores.append(r_precision(rels, R))

    print("\n==== Evaluation Results ====")
    print(f"Queries evaluated     : {len(idxs)}")
    print(f"MAP@{args.k:<2}              : {np.mean(ap_scores):.4f}")
    print(f"MRR@{args.k:<2}              : {np.mean(rr_scores):.4f}")
    print(f"Mean R-Precision      : {np.mean(rp_scores):.4f}")


if __name__ == "__main__":
    main()
