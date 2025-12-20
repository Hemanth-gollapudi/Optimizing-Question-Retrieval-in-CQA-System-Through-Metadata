# Configuration Guide

All project constants are now centralized in configuration files for easy management.

## Configuration Files

### 1. Backend Configuration (`config.py`)
**Location:** Project root

Contains all Python backend constants:
- Server ports and hosts
- API endpoints
- Ollama configuration
- Retrieval parameters
- File paths
- LLM settings

**Key Constants:**
```python
BACKEND_PORT = 5000
FRONTEND_PORT = 3000
OLLAMA_PORT = 11434
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_K = 30
FINAL_K = 2
OOD_THRESHOLD = 0.55
```

### 2. Frontend Configuration (`frontend/src/config.js`)
**Location:** `frontend/src/config.js`

Contains all React frontend constants:
- API endpoints
- Frontend port
- UI dimensions
- Colors and styling

**Key Constants:**
```javascript
API_ENDPOINT = 'http://localhost:5000/api/chat'
FRONTEND_PORT = 3000
```

## How to Change Ports

### Method 1: Edit Config Files (Permanent)

**Backend:**
Edit `config.py`:
```python
BACKEND_PORT = 8000  # Change from 5000 to 8000
```

**Frontend:**
Edit `frontend/src/config.js`:
```javascript
const BACKEND_PORT = '8000';  // Change from '5000' to '8000'
```

### Method 2: Environment Variables (Temporary)

**Backend:**
```bash
BACKEND_PORT=8000 python run_api.py
```

**Frontend:**
```bash
cd frontend
VITE_BACKEND_PORT=8000 npm run dev
```

**Launch Script:**
```bash
BACKEND_PORT=8000 FRONTEND_PORT=3001 ./launch.sh
```

### Method 3: Frontend .env File

Create `frontend/.env`:
```bash
VITE_BACKEND_HOST=localhost
VITE_BACKEND_PORT=8000
VITE_FRONTEND_PORT=3001
```

Then run:
```bash
cd frontend
npm run dev
```

## All Configurable Constants

### Server Configuration
- `BACKEND_PORT` - Backend API port (default: 5000)
- `BACKEND_HOST` - Backend host (default: 0.0.0.0)
- `FRONTEND_PORT` - Frontend dev server port (default: 3000)
- `FRONTEND_HOST` - Frontend host (default: localhost)

### Ollama Configuration
- `OLLAMA_HOST` - Ollama server host (default: localhost)
- `OLLAMA_PORT` - Ollama server port (default: 11434)
- `OLLAMA_CODE_MODEL` - Model for code questions (default: deepseek-coder:6.7b)
- `OLLAMA_GENERAL_MODEL` - Model for general questions (default: llama3:8b)

### Retrieval Configuration
- `EMBED_MODEL` - Embedding model (default: sentence-transformers/all-MiniLM-L6-v2)
- `CANDIDATE_K` - Number of candidates to retrieve (default: 30)
- `FINAL_K` - Number of final answers to return (default: 2)
- `OOD_THRESHOLD` - Out-of-dataset threshold (default: 0.55)

### LLM Configuration
- `LLM_MAX_TOKENS_CODE` - Max tokens for code responses (default: 700)
- `LLM_MAX_TOKENS_GENERAL` - Max tokens for general responses (default: 80)
- `LLM_TIMEOUT` - Request timeout in seconds (default: 300)

## Files Using Configuration

### Backend Files
- `src/api.py` - Uses config for all settings
- `run_api.py` - Uses config for server port/host
- `test_system.py` - Uses config for API endpoint

### Frontend Files
- `src/pages/ChatPage.jsx` - Uses API_ENDPOINT from config
- `vite.config.js` - Uses env vars for proxy settings

## Quick Reference

| Setting | Config File | Default | Env Variable |
|---------|-------------|---------|--------------|
| Backend Port | `config.py` | 5000 | `BACKEND_PORT` |
| Frontend Port | `frontend/src/config.js` | 3000 | `VITE_FRONTEND_PORT` |
| Ollama Port | `config.py` | 11434 | `OLLAMA_PORT` |
| API Endpoint | `frontend/src/config.js` | localhost:5000 | `VITE_BACKEND_HOST`, `VITE_BACKEND_PORT` |

## Example: Change Backend Port to 8000

**Option 1: Edit config.py**
```python
BACKEND_PORT = 8000
```

**Option 2: Environment variable**
```bash
BACKEND_PORT=8000 python run_api.py
```

**Option 3: Launch script**
```bash
BACKEND_PORT=8000 ./launch.sh
```

Then update frontend config or use env var:
```bash
cd frontend
VITE_BACKEND_PORT=8000 npm run dev
```

## Notes

- Environment variables take precedence over config file values
- Always restart servers after changing configuration
- Frontend uses Vite, so env vars must be prefixed with `VITE_`
- Backend config supports environment variable overrides automatically

