# Configuration Guide

This project uses centralized configuration files for easy management of ports, endpoints, and other constants.

## Configuration Files

### Backend Configuration (`config.py`)
Located in the project root, contains all Python backend constants:

```python
# Server Configuration
BACKEND_PORT = 5000
BACKEND_HOST = "0.0.0.0"
FRONTEND_PORT = 3000

# Ollama Configuration
OLLAMA_HOST = "localhost"
OLLAMA_PORT = 11434

# Retrieval Configuration
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CANDIDATE_K = 30
FINAL_K = 2
OOD_THRESHOLD = 0.55
```

### Frontend Configuration (`frontend/src/config.js`)
Contains all React frontend constants:

```javascript
// Backend API Configuration
export const API_CONFIG = {
  BASE_URL: 'http://localhost:5000',
  CHAT_ENDPOINT: '/api/chat',
  // ...
};
```

## Overriding Configuration

### Using Environment Variables

**Backend:**
```bash
# Override backend port
BACKEND_PORT=8000 python run_api.py

# Override multiple settings
BACKEND_PORT=8000 OLLAMA_PORT=11435 python run_api.py
```

**Frontend:**
```bash
# Create .env file in frontend directory
cd frontend
cp .env.example .env
# Edit .env with your values

# Or use inline
VITE_BACKEND_PORT=8000 npm run dev
```

### Using Launch Script
```bash
# Override ports when launching
BACKEND_PORT=8000 FRONTEND_PORT=3001 ./launch.sh
```

## Default Ports

- **Backend API**: `5000`
- **Frontend Dev Server**: `3000`
- **Ollama**: `11434`

## Changing Ports

### Option 1: Edit config files directly
1. Edit `config.py` for backend settings
2. Edit `frontend/src/config.js` for frontend settings

### Option 2: Use environment variables
Set environment variables before running:
```bash
export BACKEND_PORT=8000
export FRONTEND_PORT=3001
./launch.sh
```

### Option 3: Create .env file (Frontend only)
```bash
cd frontend
cp .env.example .env
# Edit .env file
npm run dev
```

## Configuration Structure

```
config.py              # Backend constants (Python)
frontend/
  src/
    config.js          # Frontend constants (JavaScript)
  .env.example         # Environment variable template
  vite.config.js       # Vite config (uses env vars)
```

## Important Notes

- Backend config (`config.py`) is imported by `src/api.py` and `run_api.py`
- Frontend config (`config.js`) is imported by React components
- Environment variables take precedence over config file values
- Always restart servers after changing configuration

