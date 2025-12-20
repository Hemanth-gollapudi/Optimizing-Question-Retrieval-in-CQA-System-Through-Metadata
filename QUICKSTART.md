# Quick Start Guide - CQA Chat System

## Prerequisites

1. Python 3.8+ installed
2. (Optional) Ollama installed and running for LLM fallback features

## Installation

```bash
# Install all dependencies
pip install -r requirements.txt
```

## Running the System

### Step 1: Start the Backend API Server

```bash
python run_api.py
```

You should see:
```
==================================================
Starting CQA Chat API Server
==================================================
API will be available at: http://localhost:5000
API docs will be available at: http://localhost:5000/docs
==================================================
[API] Loading dataset from: /path/to/data/cqa.csv
[API] Initializing retriever with 800 rows...
[API] Encoding 800 documents...
[API] Embeddings ready!
[API] Ready to serve requests!
```

**Note**: The first startup will take a minute or two to load the embedding model and encode all documents.

### Step 2: Start the Frontend

**Option 1: Development Server (Recommended)**
```bash
cd frontend
npm install
npm run dev
```
Then visit `http://localhost:3000`

**Option 2: Production Build**
```bash
cd frontend
npm install
npm run build
```
Then visit `http://localhost:5000` (backend serves the built frontend)

### Step 3: Start Chatting!

Type a question in the chat interface and get instant answers!

## API Endpoints

- **POST `/api/chat`**: Main chat endpoint
  - Request: `{"query": "your question here"}`
  - Response: `{"response": "answer", "source": "dataset|llm", "similarity_score": 0.85}`

- **GET `/`**: Health check and status
- **GET `/health`**: Simple health check
- **GET `/docs`**: Interactive API documentation (Swagger UI)
- **GET `/redoc`**: Alternative API documentation (ReDoc)

## How It Works

1. **Dataset Search**: When you ask a question, the system searches the CQA dataset using semantic similarity
2. **Similarity Threshold**: If the best match has similarity ≥ 0.55, it returns the answer from the dataset
3. **LLM Fallback**: If similarity < 0.55 (out-of-dataset question), it uses Ollama LLM:
   - Code questions → `deepseek-coder:6.7b`
   - General questions → `llama3:8b`

## Troubleshooting

### "Ollama is not running" error
- Install Ollama from https://ollama.ai/
- Pull required models:
  ```bash
  ollama pull deepseek-coder:6.7b
  ollama pull llama3:8b
  ```
- Make sure Ollama is running on `http://localhost:11434`

### "Dataset not found" error
- Make sure `data/cqa.csv` exists
- If not, generate it: `python src/make_data.py --out data/cqa.csv --n_per_cat 200`

### Frontend can't connect
- Make sure the API server is running on port 5000
- Check browser console for CORS errors (shouldn't happen, CORS is enabled)
- Verify the API endpoint in `frontend/src/config.js` matches your server URL
- If using dev server, make sure Vite proxy is configured correctly in `vite.config.js`

## Testing the API

You can test the API using curl:

```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How to fix WiFi disconnecting?"}'
```

Or use the interactive docs at `http://localhost:5000/docs`

