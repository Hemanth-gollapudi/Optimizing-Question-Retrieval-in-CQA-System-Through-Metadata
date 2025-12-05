# import argparse
# import re
# import numpy as np
# import pandas as pd
# import torch

# from sklearn.preprocessing import normalize
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Local embedding model (no API keys)
# from sentence_transformers import SentenceTransformer


# # -------------------------
# # Utilities
# # -------------------------
# def clean_text(s: str) -> str:
#     s = (s or "").strip()
#     s = re.sub(r"\s+", " ", s)
#     return s


# def answer_key(s: str) -> str:
#     """Normalize answer for de-dup checks."""
#     s = clean_text(s).lower()
#     s = re.sub(r"[\W_]+", "", s)  # remove punctuation/underscores/spaces
#     return s


# # -------------------------
# # Local Generator (fallback)
# # -------------------------
# class LocalGenerator:
#     def __init__(self, model_name: str = "google/flan-t5-large", device: str = "cpu"):
#         self.device = torch.device(device)
#         self.tok = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
#         self.model.eval()

#     def _detect_lang(self, question: str) -> str:
#         q = question.lower()
#         if "javascript" in q or "js" in q or "node" in q:
#             return "javascript"
#         return "python"

#     def _template(self, lang: str) -> str:
#         if lang == "javascript":
#             return (
#                 "const fs = require('fs');\n"
#                 "const input = fs.readFileSync(0, 'utf8').trim().split(/\\s+/).filter(Boolean);\n"
#                 "const a = Number(input[0] ?? 0);\n"
#                 "const b = Number(input[1] ?? 0);\n"
#                 "console.log(a + b);\n"
#             )
#         # python
#         return (
#             "a = int(input())\n"
#             "b = int(input())\n"
#             "print(a + b)\n"
#         )

#     def _looks_bad(self, text: str, lang: str) -> bool:
#         t = text.strip()

#         # too short or empty
#         if len(t) < 20:
#             return True

#         # repetition heuristic: same token repeated too much
#         toks = re.findall(r"[A-Za-z_()]+", t)
#         if toks:
#             top_freq = max(toks.count(x) for x in set(toks))
#             if top_freq / max(1, len(toks)) > 0.18:  # high repetition
#                 return True

#         # language mismatch heuristics
#         if lang == "javascript":
#             # If it contains python-only patterns, reject
#             if "input(" in t or "print(" in t or "def " in t:
#                 return True
#         else:
#             # python expected; reject obvious JS/Node boilerplate if desired
#             if "require(" in t or "console.log" in t:
#                 return True

#         return False

#     @torch.no_grad()
#     def generate(self, question: str) -> str:
#         lang = self._detect_lang(question)

#         # VERY strict prompt
#         if lang == "javascript":
#             prompt = (
#                 "Write ONLY JavaScript (Node.js) code.\n"
#                 "Task: Add two numbers.\n"
#                 "Input: two numbers from STDIN separated by space or newline.\n"
#                 "Output: their sum.\n"
#                 "Rules: Output ONLY code. No explanation. No Python.\n\n"
#                 "JavaScript code:\n"
#             )
#         else:
#             prompt = (
#                 "Write ONLY Python 3 code.\n"
#                 "Task: Add two numbers.\n"
#                 "Input: two numbers from STDIN.\n"
#                 "Output: their sum.\n"
#                 "Rules: Output ONLY code. No explanation.\n\n"
#                 "Python code:\n"
#             )

#         enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)

#         out = self.model.generate(
#             **enc,
#             max_new_tokens=120,
#             min_new_tokens=40,
#             num_beams=4,
#             do_sample=False,
#             repetition_penalty=1.25,
#             no_repeat_ngram_size=4,
#             early_stopping=True,
#         )

#         text = self.tok.decode(out[0], skip_special_tokens=True).strip()

#         # If it looks wrong/garbage, return a deterministic correct template
#         if self._looks_bad(text, lang):
#             return self._template(lang)

#         return text

# # -------------------------
# # Retriever (better embeddings)
# # -------------------------
# class SemanticRetriever:
#     def __init__(self, df: pd.DataFrame, embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
#         self.df = df.copy()

#         # Basic schema tolerance
#         # Expected columns: subject, question, answer, category, qid
#         # We’ll create best-effort fields.
#         cols = {c.lower(): c for c in self.df.columns}
#         self.col_qid = cols.get("qid", None)
#         self.col_cat = cols.get("category", None)
#         self.col_sub = cols.get("subject", cols.get("query_subject", None))
#         self.col_body = cols.get("question", cols.get("query_body", cols.get("body", None)))
#         self.col_ans = cols.get("answer", cols.get("response", None))

#         if self.col_sub is None and self.col_body is None:
#             raise ValueError("CSV must contain at least a 'subject' or 'question' column.")
#         if self.col_ans is None:
#             raise ValueError("CSV must contain an 'answer' column.")

#         # Build text used for retrieval
#         sub = self.df[self.col_sub].fillna("") if self.col_sub else ""
#         body = self.df[self.col_body].fillna("") if self.col_body else ""
#         self.df["_retrieval_text"] = (sub.astype(str) + " " + body.astype(str)).map(clean_text)

#         # Embed all dataset rows once
#         self.embedder = SentenceTransformer(embed_model)
#         emb = self.embedder.encode(self.df["_retrieval_text"].tolist(), batch_size=64, show_progress_bar=True)
#         self.emb = normalize(np.asarray(emb), norm="l2")  # cosine via dot product

#     def search(self, query: str, top_k: int = 30) -> pd.DataFrame:
#         q_emb = self.embedder.encode([query], show_progress_bar=False)
#         q_emb = normalize(np.asarray(q_emb), norm="l2")
#         scores = (self.emb @ q_emb.T).reshape(-1)  # cosine similarities

#         idx = np.argsort(-scores)[:top_k]
#         out = self.df.iloc[idx].copy()
#         out["score"] = scores[idx]
#         return out


# def print_retrieval(results: pd.DataFrame, top_n: int = 2, unique_answers: bool = True):
#     print("\n=============================================")
#     print(" Top Retrieved Results (Unique & Readable)")
#     print("=============================================\n")

#     shown = 0
#     seen = set()

#     for _, row in results.iterrows():
#         ans = str(row.get("answer") if "answer" in row else row.get("Answer", ""))  # safety
#         if "answer" not in row:
#             # if original column is not literally 'answer', use df’s stored answer col
#             # this branch usually won’t happen due to how we construct df
#             pass

#         akey = answer_key(ans)
#         if unique_answers and akey in seen:
#             continue
#         seen.add(akey)

#         qid = row.get("qid", row.get("QID", "-"))
#         cat = row.get("category", row.get("Category", "-"))
#         subj = row.get("subject", row.get("Subject", "-"))
#         score = float(row["score"])

#         print(f"[{shown+1}] QID: {qid} | Category: {cat} | Score: {score:.4f}")
#         print(f"Subject: {subj}")
#         print(f"Answer : {clean_text(ans)}")
#         print("-" * 60)

#         shown += 1
#         if shown >= top_n:
#             break


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_csv", required=True, help="Path to cqa.csv")
#     ap.add_argument("--query_subject", default="", help="Query subject/title")
#     ap.add_argument("--query_body", default="", help="Query body/details")
#     ap.add_argument("--candidate_k", type=int, default=30, help="How many retrieval candidates to consider")
#     ap.add_argument("--final_k", type=int, default=2, help="How many answers to show (top unique)")
#     ap.add_argument("--allow_duplicate_answers", action="store_true", help="Allow duplicate answers in output")
#     ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
#     ap.add_argument("--gen_model", default="google/flan-t5-base")
#     ap.add_argument("--device", default="cpu")
#     ap.add_argument(
#         "--ood_threshold",
#         type=float,
#         default=0.55,
#         help="If best cosine similarity is below this, treat as out-of-domain and use generator fallback",
#     )
#     args = ap.parse_args()

#     print(f"[Info] device={args.device}")

#     df = pd.read_csv(args.data_csv)
#     # Normalize common column names to expected ones if needed
#     # We'll rename to: qid, category, subject, question, answer when possible
#     ren = {}
#     for c in df.columns:
#         cl = c.lower()
#         if cl in ["qid", "id", "question_id"]:
#             ren[c] = "qid"
#         elif cl in ["category", "label", "class"]:
#             ren[c] = "category"
#         elif cl in ["subject", "title"]:
#             ren[c] = "subject"
#         elif cl in ["question", "body", "query_body", "text"]:
#             ren[c] = "question"
#         elif cl in ["answer", "response", "completion"]:
#             ren[c] = "answer"
#     df = df.rename(columns=ren)

#     retriever = SemanticRetriever(df, embed_model=args.embed_model)
#     generator = LocalGenerator(model_name=args.gen_model, device=args.device)

#     query = clean_text(args.query_subject + " " + args.query_body)

#     results = retriever.search(query, top_k=args.candidate_k)
#     best = float(results.iloc[0]["score"]) if not results.empty else -1.0

#     # OOD gate: don't force irrelevant dataset matches
#     if best < args.ood_threshold:
#         print("\n=============================================")
#         print(" Fallback (Out-of-dataset question detected)")
#         print("=============================================\n")
#         print(f"[Debug] best_similarity={best:.4f} < threshold={args.ood_threshold:.2f}")
#         ans = generator.generate(query)
#         print("Answer:\n" + ans)

#         return

#     # Top-2 unique answers output
#     print_retrieval(
#         results=results,
#         top_n=args.final_k,
#         unique_answers=(not args.allow_duplicate_answers),
#     )


# if __name__ == "__main__":
#     main()

import argparse
import re
import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
)


# -------------------------
# Utilities
# -------------------------
def clean_text(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def answer_key(s: str) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"[\W_]+", "", s)
    return s


def detect_language(question: str) -> str:
    q = question.lower()
    # order matters: javascript contains java
    if re.search(r"\bjavascript\b", q) or re.search(r"\bnode\.?js\b", q) or re.search(r"\bjs\b", q):
        return "JavaScript"
    if re.search(r"\bjava\b", q):
        return "Java"
    if re.search(r"\bpython\b", q) or re.search(r"\bpy\b", q):
        return "Python"
    if re.search(r"\btypescript\b", q):
        return "TypeScript"
    if re.search(r"\bc\+\+\b", q) or re.search(r"\bcpp\b", q):
        return "C++"
    if re.search(r"\bc#\b", q) or re.search(r"\bcsharp\b", q):
        return "C#"
    if re.search(r"\bgo(lang)?\b", q):
        return "Go"
    if re.search(r"\brust\b", q):
        return "Rust"
    if re.search(r"\bkotlin\b", q):
        return "Kotlin"
    if re.search(r"\bswift\b", q):
        return "Swift"
    return "Python"


def extract_code(text: str) -> str:
    t = text.strip()
    t = re.sub(r"^(answer|code)\s*:\s*", "", t, flags=re.IGNORECASE).strip()
    m = re.search(r"```(?:\w+)?\s*(.*?)```", t, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


def too_repetitive(text: str) -> bool:
    t = text.strip()
    if len(t) < 20:
        return True
    toks = re.findall(r"[A-Za-z_]+|\d+|==|!=|<=|>=|[(){}.;=+\-/*]", t)
    if not toks:
        return True
    counts = {}
    for tok in toks:
        counts[tok] = counts.get(tok, 0) + 1
    top_freq = max(counts.values())
    return (top_freq / max(1, len(toks))) > 0.18


def looks_like_wrong_language(code: str, lang: str) -> bool:
    t = code.strip()
    if len(t) < 20 or too_repetitive(t):
        return True

    python_markers = ["def ", "print(", "input(", "elif ", "None", "True", "False", "import ", "__name__"]
    js_markers = ["console.log", "require(", "let ", "const ", "function ", "process.stdin"]
    java_markers = ["public class", "static void main", "System.out", "Scanner", "BufferedReader", "InputStreamReader"]

    if lang == "Java":
        if any(m in t for m in python_markers) or any(m in t for m in js_markers):
            return True
    elif lang == "JavaScript":
        if any(m in t for m in python_markers) or any(m in t for m in java_markers):
            return True
    elif lang == "Python":
        if any(m in t for m in java_markers) or any(m in t for m in js_markers):
            return True

    return False


# -------------------------
# Retriever
# -------------------------
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

    def search(self, query: str, top_k: int) -> pd.DataFrame:
        q_emb = self.embedder.encode([query], show_progress_bar=False)
        q_emb = normalize(np.asarray(q_emb), norm="l2")
        scores = (self.emb @ q_emb.T).reshape(-1)

        idx = np.argsort(-scores)[:top_k]
        out = self.df.iloc[idx].copy()
        out["score"] = scores[idx]
        return out


def print_retrieval(results: pd.DataFrame, top_n: int = 2, unique_answers: bool = True):
    print("\n=============================================")
    print(" Top Retrieved Results (Unique & Readable)")
    print("=============================================\n")

    shown = 0
    seen = set()

    for _, row in results.iterrows():
        ans = str(row.get("answer", ""))
        akey = answer_key(ans)

        if unique_answers and akey in seen:
            continue
        seen.add(akey)

        qid = row.get("qid", "-")
        cat = row.get("category", "-")
        subj = row.get("subject", "-")
        score = float(row["score"])

        print(f"[{shown+1}] QID: {qid} | Category: {cat} | Score: {score:.4f}")
        print(f"Subject: {subj}")
        print(f"Answer : {clean_text(ans)}")
        print("-" * 60)

        shown += 1
        if shown >= top_n:
            break


# -------------------------
# Unified local generator (supports Seq2Seq AND Causal)
# -------------------------
class UnifiedLocalGenerator:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model_name = model_name

        cfg = AutoConfig.from_pretrained(model_name)
        self.is_encoder_decoder = bool(getattr(cfg, "is_encoder_decoder", False))

        self.tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)

        if self.is_encoder_decoder:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        else:
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def _generate_seq2seq(self, prompt: str) -> str:
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=384).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=180,
            min_new_tokens=60,
            num_beams=4,
            do_sample=False,
            repetition_penalty=1.25,
            no_repeat_ngram_size=4,
            length_penalty=1.05,
            early_stopping=True,
        )
        return extract_code(self.tok.decode(out[0], skip_special_tokens=True))

    @torch.no_grad()
    def _generate_causal(self, prompt: str) -> str:
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        out = self.model.generate(
            **enc,
            max_new_tokens=220,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.pad_token_id,
        )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        if text.startswith(prompt):
            text = text[len(prompt):]
        return extract_code(text)

    def _gen_once(self, prompt: str) -> str:
        if self.is_encoder_decoder:
            return self._generate_seq2seq(prompt)
        return self._generate_causal(prompt)

    def answer(self, question: str) -> str:
        lang = detect_language(question)

        # Strict, but model-agnostic instruction
        base_rules = (
            f"Write ONLY {lang} code.\n"
            "Output ONLY code. No explanation. No markdown.\n"
            "Read input from STDIN and write output to STDOUT.\n"
            "Keep it minimal and correct.\n"
        )

        prompt1 = base_rules + "\nTask:\n" + question + "\n\nCode:\n"
        code1 = self._gen_once(prompt1)

        if looks_like_wrong_language(code1, lang):
            prompt2 = (
                base_rules
                + "Do NOT output any other language.\n"
                + "Avoid repeating lines. Stop after the complete solution.\n\n"
                + "Task:\n" + question + "\n\nCode:\n"
            )
            code2 = self._gen_once(prompt2)
            return code2

        return code1


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--query_subject", default="")
    ap.add_argument("--query_body", default="")

    ap.add_argument("--candidate_k", type=int, default=30)
    ap.add_argument("--final_k", type=int, default=2)
    ap.add_argument("--allow_duplicate_answers", action="store_true")

    ap.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")

    # You can pass EITHER:
    # - google/flan-t5-large (seq2seq; weaker for code)
    # - Qwen/Qwen2.5-Coder-0.5B-Instruct or 1.5B-Instruct (causal; better for code)
    ap.add_argument("--gen_model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ood_threshold", type=float, default=0.55)
    args = ap.parse_args()

    print(f"[Info] device={args.device}")

    df = pd.read_csv(args.data_csv)

    # Normalize likely column names
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
    df = df.rename(columns=ren)

    retriever = SemanticRetriever(df, embed_model=args.embed_model)
    generator = UnifiedLocalGenerator(model_name=args.gen_model, device=args.device)

    query = clean_text(args.query_subject + " " + args.query_body)

    results = retriever.search(query, top_k=args.candidate_k)
    best = float(results.iloc[0]["score"]) if not results.empty else -1.0

    if best < args.ood_threshold:
        print("\n=============================================")
        print(" Fallback (Out-of-dataset question detected)")
        print("=============================================\n")
        print(f"[Debug] best_similarity={best:.4f} < threshold={args.ood_threshold:.2f}")
        ans = generator.answer(query)
        print("Answer:\n" + ans)
        return

    print_retrieval(
        results=results,
        top_n=args.final_k,
        unique_answers=(not args.allow_duplicate_answers),
    )


if __name__ == "__main__":
    main()
