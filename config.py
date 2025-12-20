"""
Project-wide configuration constants
Centralized configuration for backend and shared settings
"""
import os

# =========================================================
# Server Configuration
# =========================================================
# Allow environment variable overrides
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "5000"))
BACKEND_HOST = os.getenv("BACKEND_HOST", "0.0.0.0")
FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "3000"))
FRONTEND_HOST = os.getenv("FRONTEND_HOST", "localhost")

# =========================================================
# API Configuration
# =========================================================
API_ENDPOINT = f"http://localhost:{BACKEND_PORT}/api/chat"
API_DOCS_URL = f"http://localhost:{BACKEND_PORT}/docs"

# =========================================================
# Ollama Configuration
# =========================================================
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

# Models
OLLAMA_CODE_MODEL = "llama3:8b"
OLLAMA_GENERAL_MODEL = "llama3:8b"

# =========================================================
# Retrieval Configuration
# =========================================================
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_K = 30
FINAL_K = 2
OOD_THRESHOLD = 0.55

# =========================================================
# File Paths
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_CSV = os.path.join(DATA_DIR, "cqa.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
FRONTEND_BUILD_DIR = os.path.join(BASE_DIR, "frontend", "dist")

# =========================================================
# LLM Configuration
# =========================================================
LLM_MAX_TOKENS_CODE = 700
LLM_MAX_TOKENS_GENERAL = 80
LLM_TIMEOUT = 300

# Rebuild URLs after environment variable overrides
API_ENDPOINT = f"http://localhost:{BACKEND_PORT}/api/chat"
API_DOCS_URL = f"http://localhost:{BACKEND_PORT}/docs"
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
OLLAMA_GENERATE_ENDPOINT = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_ENDPOINT = f"{OLLAMA_BASE_URL}/api/tags"

