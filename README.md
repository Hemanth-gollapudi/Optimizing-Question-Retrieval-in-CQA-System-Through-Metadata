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
```bash
pip install -r requirements.txt
