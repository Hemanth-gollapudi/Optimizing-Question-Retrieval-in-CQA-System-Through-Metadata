# Minor Project: CQA Question Retrieval using Category Prediction + Query Expansion (ITRLM-style)

## Overview
This project demonstrates a Community Question Answering retrieval pipeline that addresses the lexical gap problem.
It implements:
1. Transformer-based category prediction (BERT/DistilBERT).
2. Category-specific translation dictionary built from metadata (subject + answer).
3. Query expansion using a lightweight RAG-style generator (optional).
4. Retrieval and ranking using BM25 + metadata/category boosts.

## Folder Structure
- `src/` : source code
- `data/` : dataset (`cqa.csv`)
- `artifacts/` : saved models (category model + translation dictionary)

## Setup

Step 1: Install requirements
pip install -r requirements.txt

Step 2: Generate synthetic dataset
python src/make_data.py --out data/cqa.csv --n_per_cat 200

Step 3: Train category classifier + dictionaries
python src/train.py --data_csv data/cqa.csv

Step 4: Run retrieval demo
python src/demo.py --data_csv data/cqa.csv --query_subject "give your subject" --query_body "your question"

Optional: enable RAG-style query expansion
python src/demo.py --data_csv data/cqa.csv --use_expansion

Step 5: Evaluate
python src/evaluate.py --data_csv data/cqa.csv --k 10 --num_queries 200

Optional: enable RAG-style query expansion
python src/evaluate.py --data_csv data/cqa.csv --k 10 --num_queries 200 --use_expansion