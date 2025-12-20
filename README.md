# Minor Project: CQA Question Retrieval using Category Prediction + Query Expansion (ITRLM-style)

## Overview
This project demonstrates a Community Question Answering retrieval pipeline that addresses the lexical gap problem.
It implements:
1. Transformer-based category prediction (BERT/DistilBERT).
2. Category-specific translation dictionary built from metadata (subject + answer).
3. Query expansion using a lightweight RAG-style generator (optional).
4. Retrieval using semantic vector retrieval with SentenceTransformer embeddings + cosine similarity.

## Models Used

1. **Code-related queries**: `deepseek-coder:6.7b`
2. **General queries**: `llama3:8b` 

## Folder Structure
- `src/` : source code
- `data/` : dataset (`cqa.csv`)
- `artifacts/` : saved models (category model + translation dictionary)
- `Create` : category_model and translation_dict folders in artifacts

## Setup
```bash
Step 1: Install requirements
pip install -r requirements.txt

Step 2: Generate synthetic dataset (if not already present)
python src/make_data.py --out data/cqa.csv --n_per_cat 200

Step 3: Train category classifier + dictionaries (optional, for advanced features)
python src/train.py --data_csv data/cqa.csv

Step 4: Run retrieval demo (CLI version)
python src/demo.py --data_csv data/cqa.csv --query_body "Java code for addition of numbers"

Step 5: Evaluate
python src/evaluate.py --data_csv data/cqa.csv --k 10 --num_queries 200 --exclude_self --relevance answer
```

## Running the Web Interface

### Quick Start (Recommended)
```bash
# Launch both backend and frontend with one command
./launch.sh
```

This will start:
- **Backend API**: `http://localhost:5000`
- **React Frontend**: `http://localhost:3000` (development mode)

### Manual Launch

#### Option 1: React Frontend (Recommended - Modular & Scrollable)
```bash
# Terminal 1: Start backend
source venv/bin/activate
python run_api.py

# Terminal 2: Start React frontend
cd frontend
npm install  # First time only
npm run dev
```

Visit `http://localhost:3000` for the React frontend.

#### Option 2: Simple HTML Frontend
```bash
# Start backend
source venv/bin/activate
python run_api.py
```

Visit `http://localhost:5000` - the backend serves the React frontend build automatically (if built), or use the dev server at `http://localhost:3000`.

### API Endpoints
- **Frontend Interface**: `http://localhost:5000/` (serves HTML)
- **API Endpoint**: `http://localhost:5000/api/chat`
- **API Documentation**: `http://localhost:5000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:5000/redoc` (ReDoc)
- **Health Check**: `http://localhost:5000/health`

**Frontend Features:**
- ✅ **React-based modular architecture** - Easy to extend and maintain
- ✅ **Fully scrollable** - Fixed scrolling issues
- ✅ Modern chat interface with message history sidebar
- ✅ Real-time API integration
- ✅ Responsive design with gradient themes
- ✅ Sidebar toggle for more screen space

### Prerequisites for LLM Fallback
If you want to use the LLM fallback feature (for out-of-dataset questions):
- Install and run [Ollama](https://ollama.ai/)
- Pull the required models:
  ```bash
  ollama pull deepseek-coder:6.7b
  ollama pull llama3:8b
  ```
- Make sure Ollama is running on `http://localhost:11434`

**Note**: The system will work without Ollama, but will only return answers from the dataset. Questions with low similarity scores (< 0.55) will return an error if Ollama is not available.

## Testing

To run the test suite:
```bash
python test_system.py
```

The test script will:
- Test various question categories and topics
- Measure response times and similarity scores
- Generate a detailed `test_results.json` report
- Use a 180-second timeout to accommodate LLM responses

**Note**: Make sure the API server is running before executing tests.