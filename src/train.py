import os
import re
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

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

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class CategoryDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int):
        self.texts, self.labels, self.tokenizer, self.max_len = texts, labels, tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, max_length=self.max_len, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

@dataclass
class CatModelConfig:
    model_name: str = "distilbert-base-uncased"
    max_len: int = 256
    lr: float = 2e-5
    batch_size: int = 16
    epochs: int = 2
    warmup_ratio: float = 0.1

class CategoryPredictor:
    def __init__(self, cfg: CatModelConfig, device: str):
        self.cfg = cfg
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(cfg.model_name)
        self.model = None
        self.le = LabelEncoder()

    def fit(self, df: pd.DataFrame, save_dir: str):
        df = df.copy()
        df["text"] = (df["subject"].fillna("") + " [SEP] " + df["body"].fillna("")).map(clean_text)

        y = self.le.fit_transform(df["category"].astype(str).values)
        X_train, X_val, y_train, y_val = train_test_split(
            df["text"].tolist(), y.tolist(), test_size=0.2, random_state=42, stratify=y
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.cfg.model_name, num_labels=len(self.le.classes_)
        ).to(self.device)

        train_ds = CategoryDataset(X_train, y_train, self.tok, self.cfg.max_len)
        val_ds = CategoryDataset(X_val, y_val, self.tok, self.cfg.max_len)

        collator = DataCollatorWithPadding(self.tok)
        train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=collator)
        val_loader = DataLoader(val_ds, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=collator)

        opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        total_steps = self.cfg.epochs * len(train_loader)
        warmup_steps = int(self.cfg.warmup_ratio * total_steps)
        sched = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

        best_acc = -1.0
        for ep in range(self.cfg.epochs):
            self.model.train()
            for batch in tqdm(train_loader, desc=f"[Cat] train {ep+1}/{self.cfg.epochs}"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                loss = self.model(**batch).loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()

            self.model.eval()
            preds, gold = [], []
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"[Cat] val {ep+1}/{self.cfg.epochs}"):
                    gold.extend(batch["labels"].numpy().tolist())
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    logits = self.model(**batch).logits
                    preds.extend(logits.argmax(-1).cpu().numpy().tolist())

            acc = accuracy_score(gold, preds)
            print(f"[Cat] epoch={ep+1} val_acc={acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                self.save(save_dir)

        print(f"[Cat] best_val_acc={best_acc:.4f}")

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)
        with open(os.path.join(path, "label_encoder.json"), "w", encoding="utf-8") as f:
            json.dump(self.le.classes_.tolist(), f, indent=2)

def build_translation_dict(df: pd.DataFrame, min_count: int = 2, smooth: float = 1.0) -> Dict[str, Dict[str, Dict[str, float]]]:
    df = df.copy()
    df["q_text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).map(clean_text)
    df["meta_text"] = (df["subject"].fillna("") + " " + df["answer"].fillna("")).map(clean_text)

    dicts = {}
    for cat, g in tqdm(df.groupby("category"), desc="[Dict] building"):
        src_counts, tgt_counts, co = {}, {}, {}
        for _, row in g.iterrows():
            q = set(normalize_tokens(row["q_text"]))
            m = set(normalize_tokens(row["meta_text"]))
            for s in q:
                src_counts[s] = src_counts.get(s, 0) + 1
            for t in m:
                tgt_counts[t] = tgt_counts.get(t, 0) + 1
            for s in q:
                for t in m:
                    co[(s, t)] = co.get((s, t), 0) + 1

        cat_dict = {}
        for (s, t), cst in co.items():
            if src_counts.get(s, 0) < min_count or tgt_counts.get(t, 0) < min_count:
                continue
            cat_dict.setdefault(s, {})
            cat_dict[s][t] = (cst + smooth) / (src_counts[s] + smooth)

        for s in list(cat_dict.keys()):
            total = sum(cat_dict[s].values())
            if total <= 0:
                del cat_dict[s]
                continue
            for t in list(cat_dict[s].keys()):
                cat_dict[s][t] /= total

        dicts[str(cat)] = cat_dict

    return dicts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=str, default="data/cqa.csv")
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--cat_model", type=str, default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--dict_min_count", type=int, default=2)
    args = ap.parse_args()

    # Make output paths absolute (anchored at project root = current working dir)
    out_dir = os.path.abspath(args.out_dir)
    cat_dir = os.path.join(out_dir, "category_model")
    dict_dir = os.path.join(out_dir, "translation_dict")
    os.makedirs(cat_dir, exist_ok=True)
    os.makedirs(dict_dir, exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"[Info] device={device}")
    print(f"[Info] Saving artifacts to: {out_dir}")

    df = pd.read_csv(args.data_csv)
    needed = ["qid", "subject", "body", "category", "answer"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing column in CSV: {col}")

    # Train category model
    cat = CategoryPredictor(
        CatModelConfig(
            model_name=args.cat_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        ),
        device=device,
    )
    cat.fit(df, save_dir=cat_dir)

    # Build translation dict
    trans = build_translation_dict(df, min_count=args.dict_min_count)
    out_path = os.path.join(dict_dir, "cat_translation_dict.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trans, f)
    print(f"[OK] Translation dictionary saved to: {out_path}")

if __name__ == "__main__":
    main()
