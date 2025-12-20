#!/usr/bin/env python3
"""
Server startup script for ITRLM+RAG API

Usage:
    python run_server.py
    python run_server.py --port 8080
    python run_server.py --host 0.0.0.0 --port 8000 --reload
"""

import argparse
import uvicorn
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run ITRLM+RAG FastAPI Server")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development mode)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Starting ITRLM+RAG FastAPI Server")
    print("=" * 60)
    print(f"📍 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🔄 Auto-reload: {'Enabled' if args.reload else 'Disabled'}")
    print(f"👷 Workers: {args.workers}")
    print("=" * 60)
    print(f"\n📖 API Documentation:")
    print(f"   Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"   ReDoc:      http://{args.host}:{args.port}/redoc")
    print("\n" + "=" * 60 + "\n")
    
    # Run the server
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1  # workers > 1 not compatible with reload
    )


if __name__ == "__main__":
    main()

