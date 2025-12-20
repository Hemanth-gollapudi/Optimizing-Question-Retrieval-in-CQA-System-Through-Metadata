import os
import re
import sys
import json
import time
import subprocess
import asyncio
import numpy as np
import pandas as pd
import requests
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Import project configuration
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# =========================================================
# Utilities (from demo.py)
# =========================================================

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def answer_key(s: str) -> str:
    return re.sub(r"[\W_]+", "", clean_text(s).lower())


def is_code_question(text: str) -> bool:
    keywords = [
        "code", "program", "write", "implement",
        "algorithm", "function", "class",
        "java", "python", "c++", "javascript", "js"
    ]
    t = text.lower()
    return any(k in t for k in keywords)


def strip_code_fences(text: str) -> str:
    text = re.sub(r"```[a-zA-Z]*", "", text)
    return text.strip("`\n ")


# =========================================================
# Ollama Client (from demo.py)
# =========================================================

def ollama_generate(model: str, prompt: str, max_tokens: int = 512) -> str:
    try:
        r = requests.post(
            OLLAMA_GENERATE_ENDPOINT,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens
                }
            },
            timeout=LLM_TIMEOUT
        )
        r.raise_for_status()
        return r.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama is not running. Please start Ollama and ensure the models ({OLLAMA_CODE_MODEL}, {OLLAMA_GENERAL_MODEL}) are available."
        )
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 404:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model}' not found. Please download it using: ollama pull {model}"
            )
        raise HTTPException(status_code=500, detail=f"Ollama HTTP error: {str(e)}")
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 404:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{model}' not found. Please download it using: ollama pull {model}"
            )
        raise HTTPException(status_code=500, detail=f"Ollama HTTP error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")


# =========================================================
# Semantic Retriever (from demo.py)
# =========================================================

class SemanticRetriever:
    def __init__(self, df: pd.DataFrame, embed_model: str):
        self.df = df.copy()

        text = (
            self.df.get("subject", "").fillna("").astype(str)
            + " "
            + self.df.get("question", "").fillna("").astype(str)
        ).map(clean_text)

        self.embedder = SentenceTransformer(embed_model)
        print(f"[API] Encoding {len(text)} documents...")
        emb = self.embedder.encode(text.tolist(), batch_size=64, show_progress_bar=True)
        self.emb = normalize(np.asarray(emb), norm="l2")
        print("[API] Embeddings ready!")

    def search(self, query: str, k: int):
        # Optimize: use convert_to_numpy and disable progress for faster encoding
        q_emb = self.embedder.encode(
            [query], 
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False  # We normalize manually for consistency
        )
        q_emb = normalize(np.asarray(q_emb), norm="l2")
        scores = (self.emb @ q_emb.T).reshape(-1)

        idx = np.argsort(-scores)[:k]
        out = self.df.iloc[idx].copy()
        out["score"] = scores[idx]
        return out


# =========================================================
# Configuration (imported from config.py)
# =========================================================
BASE_DIR = config.BASE_DIR
DATA_CSV = config.DATA_CSV
FRONTEND_BUILD_DIR = config.FRONTEND_BUILD_DIR
EMBED_MODEL = config.EMBED_MODEL
CANDIDATE_K = config.CANDIDATE_K
FINAL_K = config.FINAL_K
OOD_THRESHOLD = config.OOD_THRESHOLD
OLLAMA_BASE_URL = config.OLLAMA_BASE_URL
OLLAMA_GENERATE_ENDPOINT = config.OLLAMA_GENERATE_ENDPOINT
OLLAMA_CODE_MODEL = config.OLLAMA_CODE_MODEL
OLLAMA_GENERAL_MODEL = config.OLLAMA_GENERAL_MODEL
LLM_MAX_TOKENS_CODE = config.LLM_MAX_TOKENS_CODE
LLM_MAX_TOKENS_GENERAL = config.LLM_MAX_TOKENS_GENERAL
LLM_TIMEOUT = config.LLM_TIMEOUT

# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(title="CQA Chat API", version="1.0.0")

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve React build
if os.path.exists(FRONTEND_BUILD_DIR):
    app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_BUILD_DIR, "assets")), name="static")

# Global retriever instance
retriever: Optional[SemanticRetriever] = None
df: Optional[pd.DataFrame] = None


# =========================================================
# Request/Response Models
# =========================================================

class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    response: str
    source: str  # "dataset" or "llm"
    similarity_score: Optional[float] = None


# =========================================================
# Startup: Load dataset and initialize retriever
# =========================================================

@app.on_event("startup")
async def startup_event():
    global retriever, df
    
    print(f"[API] Loading dataset from: {DATA_CSV}")
    if not os.path.exists(DATA_CSV):
        raise FileNotFoundError(f"Dataset not found: {DATA_CSV}")
    
    df = pd.read_csv(DATA_CSV)
    
    # Normalize column names
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ["subject", "title"]:
            ren[c] = "subject"
        elif cl in ["question", "body", "text"]:
            ren[c] = "question"
        elif cl in ["answer", "response", "completion"]:
            ren[c] = "answer"
    df = df.rename(columns=ren)
    
    print(f"[API] Initializing retriever with {len(df)} rows...")
    retriever = SemanticRetriever(df, EMBED_MODEL)
    print("[API] Ready to serve requests!")


# =========================================================
# Frontend Serving (must be last to allow API routes to be matched first)
# =========================================================


# =========================================================
# Health Check
# =========================================================

@app.get("/api/status")
async def api_status():
    return {
        "status": "ok",
        "message": "CQA Chat API is running",
        "dataset_rows": len(df) if df is not None else 0
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/api/test-results")
async def get_test_results():
    """Get test results from test_results.json if it exists"""
    test_results_path = os.path.join(BASE_DIR, "test_results.json")
    
    if os.path.exists(test_results_path):
        try:
            with open(test_results_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            return {"error": f"Error reading test results: {str(e)}"}
    else:
        return {"error": "Test results not found. Run tests first."}


@app.post("/api/run-tests")
async def run_tests():
    """Trigger test execution (non-blocking)"""
    test_script = os.path.join(BASE_DIR, "test_system.py")
    
    print(f"[API] Run tests requested. Checking test script: {test_script}")
    
    if not os.path.exists(test_script):
        print(f"[API] ERROR: Test script not found at {test_script}")
        raise HTTPException(status_code=404, detail=f"Test script not found at {test_script}")
    
    print(f"[API] Starting test execution: {sys.executable} {test_script}")
    
    try:
        # Set environment variables to suppress tokenizers warnings
        env = os.environ.copy()
        env['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Run test script asynchronously to avoid blocking the event loop
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            test_script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            cwd=BASE_DIR  # Run from project root
        )
        
        print(f"[API] Test process started (PID: {process.pid})")
        
        # Wait for process with timeout (600 seconds = 10 minutes)
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=600.0
            )
            print(f"[API] Test process completed with return code: {process.returncode}")
        except asyncio.TimeoutError:
            # Kill the process if it times out
            print(f"[API] Test process timed out, killing process...")
            process.kill()
            await process.wait()
            raise HTTPException(status_code=504, detail="Test execution timed out after 10 minutes")
        
        # Decode output
        stdout_text = stdout.decode('utf-8') if stdout else ''
        stderr_text = stderr.decode('utf-8') if stderr else ''
        
        # Log output for debugging
        if stdout_text:
            print(f"[API] Test stdout (last 500 chars): {stdout_text[-500:]}")
        if stderr_text:
            print(f"[API] Test stderr (last 500 chars): {stderr_text[-500:]}")
        
        if process.returncode == 0:
            # Try to read results
            test_results_path = os.path.join(BASE_DIR, "test_results.json")
            print(f"[API] Looking for results file: {test_results_path}")
            
            if os.path.exists(test_results_path):
                print(f"[API] Reading test results from {test_results_path}")
                with open(test_results_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                print(f"[API] Successfully loaded test results")
                return {"status": "success", "results": data}
            else:
                print(f"[API] WARNING: Test results file not found at {test_results_path}")
                return {
                    "status": "success",
                    "message": "Tests completed but results file not found",
                    "stdout": stdout_text[-1000:] if len(stdout_text) > 1000 else stdout_text,
                    "stderr": stderr_text[-1000:] if len(stderr_text) > 1000 else stderr_text
                }
        else:
            print(f"[API] Test execution failed with return code {process.returncode}")
            return {
                "status": "error",
                "message": "Tests failed",
                "error": stderr_text[-2000:] if len(stderr_text) > 2000 else stderr_text,
                "stdout": stdout_text[-2000:] if len(stdout_text) > 2000 else stdout_text,
                "returncode": process.returncode
            }
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] ERROR in run_tests: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error running tests: {str(e)}")


# =========================================================
# Main Chat Endpoint
# =========================================================

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global retriever, df
    
    start_time = time.time()
    
    if retriever is None or df is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")
    
    query = clean_text(request.query)
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    # Search in dataset
    search_start = time.time()
    results = retriever.search(query, CANDIDATE_K)
    search_time = time.time() - search_start
    print(f"[API] Search completed in {search_time:.3f}s")
    
    best_score = float(results.iloc[0]["score"]) if not results.empty else -1.0
    
    # Dataset HIT: return answer from dataset
    if best_score >= OOD_THRESHOLD:
        process_start = time.time()
        seen = set()
        answers = []
        
        for _, row in results.iterrows():
            ans = str(row.get("answer", "")).strip()
            if not ans:
                continue
                
            key = answer_key(ans)
            
            # Skip if we've seen this exact normalized answer
            if key in seen:
                continue
            seen.add(key)
            
            # Also check if this answer is a substring of an already added answer
            # (to avoid duplicates like "X" and "X. If it continues...")
            is_duplicate = False
            for existing_ans in answers:
                existing_key = answer_key(existing_ans)
                # If one is a substring of the other (after normalization), skip
                if key in existing_key or existing_key in key:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                answers.append(ans)
                if len(answers) >= FINAL_K:
                    break
        
        # If we have multiple answers, join them nicely
        if len(answers) == 1:
            response_text = answers[0]
        elif len(answers) > 1:
            # Format multiple answers with clear separation
            response_text = "\n\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
        else:
            response_text = "No answer found in dataset."
        
        total_time = time.time() - start_time
        print(f"[API] Total response time: {total_time:.3f}s (search: {search_time:.3f}s, process: {time.time() - process_start:.3f}s)")
        
        return ChatResponse(
            response=response_text,
            source="dataset",
            similarity_score=best_score
        )
    
    # Fallback to LLM (OOD question)
    print(f"[API] OOD detected (similarity={best_score:.4f}), using LLM...")
    
    llm_start = time.time()
    fallback_message = "I apologize, but the model needs training on a larger dataset to answer this question. I will get back to you on that."
    
    try:
        if is_code_question(query):
            prompt = (
                "Write COMPLETE, VALID, COMPILABLE code.\n"
                "No explanation. No markdown.\n"
                "Stop after code ends.\n\n"
                f"{query}\n"
            )
            out = ollama_generate(OLLAMA_CODE_MODEL, prompt, max_tokens=LLM_MAX_TOKENS_CODE)
            response_text = strip_code_fences(out)
        else:
            prompt = (
                "Give ONLY the final factual answer.\n"
                "One short sentence.\n"
                "No explanation.\n\n"
                f"Question: {query}\n"
                "Answer:"
            )
            out = ollama_generate(OLLAMA_GENERAL_MODEL, prompt, max_tokens=LLM_MAX_TOKENS_GENERAL)
            response_text = out
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        print(f"[API] Total response time: {total_time:.3f}s (search: {search_time:.3f}s, LLM: {llm_time:.3f}s)")
        
        return ChatResponse(
            response=response_text,
            source="llm",
            similarity_score=best_score
        )
    except (HTTPException, Exception) as e:
        # Catch any Ollama errors and return a friendly fallback message
        error_msg = str(e)
        print(f"[API] Ollama error occurred: {error_msg}")
        print(f"[API] Returning fallback message for query: {query}")
        
        llm_time = time.time() - llm_start
        total_time = time.time() - start_time
        print(f"[API] Total response time: {total_time:.3f}s (search: {search_time:.3f}s, LLM failed)")
        
        return ChatResponse(
            response=fallback_message,
            source="llm",
            similarity_score=best_score
        )


# =========================================================
# Frontend Serving (catch-all route - must be last)
# =========================================================

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """
    Serve the React frontend build.
    This catch-all route handles client-side routing for React Router.
    All non-API routes will serve index.html, allowing React Router to handle routing.
    """
    # Don't serve HTML for API routes or static assets
    if full_path.startswith("api/") or full_path.startswith("static/") or full_path.startswith("docs") or full_path.startswith("openapi.json"):
        raise HTTPException(status_code=404, detail="Not found")
    
    react_index = os.path.join(FRONTEND_BUILD_DIR, "index.html")
    if os.path.exists(react_index):
        with open(react_index, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        # Provide helpful message for development
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Frontend Not Built</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
                    color: #f0f0f0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    padding: 40px;
                    background: rgba(26, 26, 46, 0.8);
                    border-radius: 20px;
                    border: 2px solid rgba(102, 126, 234, 0.3);
                    max-width: 600px;
                }
                h1 { color: #667eea; margin-bottom: 20px; }
                .code {
                    background: rgba(0, 0, 0, 0.3);
                    padding: 15px;
                    border-radius: 8px;
                    font-family: monospace;
                    margin: 20px 0;
                    color: #f093fb;
                }
                .link {
                    color: #667eea;
                    text-decoration: none;
                    font-weight: bold;
                }
                .link:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 React Frontend Not Built</h1>
                <p>The React frontend build is not available. For development, use the dev server:</p>
                <div class="code">
                    cd frontend<br>
                    npm install<br>
                    npm run dev
                </div>
                <p>Then visit <a href="http://localhost:3000" class="link">http://localhost:3000</a></p>
                <hr style="border-color: rgba(102, 126, 234, 0.3); margin: 30px 0;">
                <p><strong>Or build for production:</strong></p>
                <div class="code">
                    cd frontend<br>
                    npm install<br>
                    npm run build
                </div>
                <p style="margin-top: 20px;">
                    <a href="/docs" class="link">📚 API Documentation</a> | 
                    <a href="/api/status" class="link">🔍 API Status</a>
                </p>
            </div>
        </body>
        </html>
        """
        return HTMLResponse(content=html_content, status_code=200)

