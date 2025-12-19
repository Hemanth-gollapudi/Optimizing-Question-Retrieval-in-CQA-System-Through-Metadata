#!/usr/bin/env python3
"""
FastAPI server startup script for CQA Chat API
"""
import uvicorn
import sys
import os

# Import project configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

if __name__ == "__main__":
    print("=" * 50)
    print("Starting CQA Chat API Server")
    print("=" * 50)
    print(f"API will be available at: http://{config.BACKEND_HOST}:{config.BACKEND_PORT}")
    print(f"API docs will be available at: http://{config.BACKEND_HOST}:{config.BACKEND_PORT}/docs")
    print("=" * 50)
    
    uvicorn.run(
        "src.api:app",
        host=config.BACKEND_HOST,
        port=config.BACKEND_PORT,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

