# ITRLM+RAG Implementation

**Improved Translation-based Language Model (ITRLM)** with **Retrieval-Augmented Generation (RAG)** enhancements for question-answer similarity and ranking.

## 🚀 Quick Start

### Prerequisites

- **Apple Silicon Mac** (M1/M2/M3)
- **pyenv** installed
- **Python 3.10.14**

### Installation

```bash
# 1. Clone and enter project
cd itrlm_rag_project

# 2. Run setup script (Apple Silicon optimized)
bash setup.sh

# 3. Execute the main pipeline
python main.py
```

## 📁 Project Structure

```
itrlm_rag_project/
│
├── README.md
├── requirements.txt
├── setup.sh                # bootstrap environment setup
├── main.py                 # entry point for training & evaluation
│
├── data/
│   ├── semeval_loader.py   # SemEval-2017 dataset processing
│   ├── yahoo_loader.py     # Yahoo Answers dataset processing
│   └── __init__.py
│
├── src/
│   ├── text_processing.py  # cleaning, stopwords, stemming
│   ├── pmi_dictionary.py   # builds general + category-specific PMI dictionaries
│   ├── category_predictor.py  # fine-tuned BERT classifier
│   ├── rag_generator.py    # answer generation using RAG
│   ├── itrlm_scorer.py     # ITRLM scoring implementation
│   ├── hybrid_ranker.py    # multi-stage ranking pipeline
│   ├── evaluation.py       # MAP, MRR, Precision@k
│   ├── utils.py            # common helpers (seed, logging, paths)
│   └── __init__.py
│
├── configs/
│   ├── default.yaml        # central config (paths, params, hyperparams)
│   └── model_params.yaml   # α, β, λ, γ, etc.
│
├── notebooks/
│   └── exploration.ipynb   # exploratory analysis notebook
│
└── outputs/
    ├── checkpoints/        # model checkpoints
    ├── logs/              # training and evaluation logs
    └── results/           # evaluation results
```

## 🧠 Architecture Overview

### ITRLM (Improved Translation-based Language Model)

- **PMI Dictionaries**: General and category-specific probabilistic translation dictionaries
- **Translation Similarity**: Computes similarity using Pointwise Mutual Information (PMI)
- **Category-Aware**: Leverages category-specific translations for better accuracy

### RAG (Retrieval-Augmented Generation)

- **Knowledge Base**: Built from training data using sentence embeddings
- **Context Retrieval**: Retrieves relevant context for query expansion
- **Answer Generation**: Generates enhanced answers using retrieved context

### Hybrid Ranking Pipeline

1. **Semantic Ranking**: Sentence transformer-based similarity
2. **ITRLM Scoring**: Translation-based similarity with parameters α, β, λ, γ
3. **RAG Enhancement**: RAG-generated content for query expansion
4. **Final Ranking**: Weighted combination of all approaches

## ⚙️ Configuration

### Model Parameters (`configs/model_params.yaml`)

```yaml
model:
  # Translation-based similarity weights
  alpha: 0.3 # Weight for general PMI translations
  beta: 0.8 # Weight for category-specific PMI translations
  lambda_: 0.2 # Weight for semantic similarity
  gamma: 0.8 # Weight for RAG-generated answers

  # Translation parameters
  top_n_translations: 100
  min_pmi_threshold: 0.1
  max_translation_length: 50
```

### RAG Configuration

```yaml
rag:
  embed_model: sentence-transformers/all-mpnet-base-v2
  gen_model: mistralai/Mistral-7B-Instruct-v0.2
  top_k_ctx: 5
  max_answer_tokens: 50
  temperature: 0.7
  do_sample: true
```

## 🔧 Core Components

### 1. Text Processing (`src/text_processing.py`)

- Text cleaning and normalization
- Tokenization and stopword removal
- Stemming and vocabulary building

### 2. PMI Dictionary (`src/pmi_dictionary.py`)

- Builds general PMI dictionary from all training data
- Creates category-specific PMI dictionaries
- Provides translation-based query expansion

### 3. Category Predictor (`src/category_predictor.py`)

- Fine-tuned BERT classifier for question categorization
- Supports multiple categories (technology, science, health, etc.)
- Provides confidence scores for predictions

### 4. RAG Generator (`src/rag_generator.py`)

- Builds knowledge base from training data
- Retrieves relevant context using sentence embeddings
- Generates enhanced answers using language models

### 5. ITRLM Scorer (`src/itrlm_scorer.py`)

- Computes translation-based similarity using PMI
- Combines general and category-specific translations
- Integrates semantic similarity for robust scoring

### 6. Hybrid Ranker (`src/hybrid_ranker.py`)

- Multi-stage ranking pipeline
- Combines semantic, ITRLM, and RAG approaches
- Supports feedback-based reranking

### 7. Evaluation (`src/evaluation.py`)

- Computes MAP (Mean Average Precision)
- Calculates MRR (Mean Reciprocal Rank)
- Measures Precision@k for different k values

## 📊 Usage Examples

### Basic Usage

```python
from src.utils import set_seed
from src.text_processing import TextProcessor
from src.pmi_dictionary import PMIDictionary
from src.category_predictor import CategoryPredictor
from src.rag_generator import RAGAnswerGen
from src.itrlm_scorer import ITRLMScorer
from src.hybrid_ranker import HybridRanker

# Initialize components
text_processor = TextProcessor()
pmi_dict = PMIDictionary(config, text_processor)
cat_predictor = CategoryPredictor(config, text_processor)
rag_generator = RAGAnswerGen(config, text_processor)
itrlm_scorer = ITRLMScorer(pmi_dict, config, text_processor)
hybrid_ranker = HybridRanker(itrlm_scorer, cat_predictor, rag_generator, config)

# Build models
pmi_dict.build_all(training_data)
cat_predictor.load_or_train(training_data)
rag_generator.build_knowledge_base(training_data)

# Rank candidates
query = "What is machine learning?"
candidates = [
    {"answer": "ML is a subset of AI...", "category": "technology"},
    {"answer": "Paris is the capital...", "category": "geography"}
]

ranked_results = hybrid_ranker.rank(query, candidates, top_k=5)
```

### Evaluation

```python
from src.evaluation import Evaluator

evaluator = Evaluator(config)
results = evaluator.run(hybrid_ranker, test_data)

print(f"MAP: {results['map']:.3f}")
print(f"MRR: {results['mrr']:.3f}")
print(f"Precision@1: {results['precision@1']:.3f}")
```

## 📈 Performance Metrics

The system evaluates performance using standard information retrieval metrics:

- **MAP (Mean Average Precision)**: Overall ranking quality
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **Precision@k**: Accuracy of top-k results
- **NDCG**: Normalized Discounted Cumulative Gain

## 🔬 Research Background

### ITRLM (Improved Translation-based Language Model)

- Extends traditional translation-based similarity
- Uses PMI for probabilistic translation modeling
- Incorporates category-specific translations
- Combines with semantic similarity for robustness

### RAG Integration

- Retrieval-Augmented Generation for query expansion
- Knowledge base built from training data
- Context-aware answer generation
- Enhanced semantic understanding

### Hybrid Approach

- Multi-stage ranking pipeline
- Combines multiple similarity signals
- Weighted fusion of different approaches
- Feedback-based reranking support

## 🛠️ Development

### Adding New Datasets

1. Create a new loader in `data/` directory
2. Implement `load_data()` method
3. Return list of dictionaries with required fields
4. Update imports in `main.py`

### Extending Models

1. Add new model class in `src/` directory
2. Implement required interface methods
3. Update configuration files
4. Integrate with hybrid ranker

### Custom Evaluation

1. Extend `Evaluator` class in `src/evaluation.py`
2. Add new metric computation methods
3. Update configuration for new metrics
4. Test with existing evaluation pipeline

## 📝 Logging and Monitoring

- **Structured Logging**: All components use consistent logging
- **Progress Tracking**: Training and evaluation progress monitoring
- **Result Persistence**: Automatic saving of results and checkpoints
- **Error Handling**: Comprehensive error handling and recovery

## 🔧 Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch sizes or use smaller models
2. **Model Loading**: Ensure all dependencies are installed
3. **Data Loading**: Check data paths and formats
4. **Performance**: Adjust model parameters for your hardware

### Apple Silicon Optimization

- Uses MPS (Metal Performance Shaders) when available
- Optimized PyTorch builds for Apple Silicon
- Efficient memory management for M1/M2/M3 chips

## 📚 References

- SemEval-2017 Task 3: Question-Answer Similarity
- Yahoo Answers Dataset
- BERT: Pre-training of Deep Bidirectional Transformers
- RAG: Retrieval-Augmented Generation
- PMI-based Translation Models

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Status**: ✅ Production Ready - Full ITRLM+RAG implementation with comprehensive evaluation pipeline
