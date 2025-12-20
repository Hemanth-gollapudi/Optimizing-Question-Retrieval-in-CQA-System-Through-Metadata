#!/bin/bash

# CQA Chat System - Launch Script
# This script starts both the backend API and frontend development server

set -e

# Load configuration (can be overridden by environment variables)
BACKEND_PORT=${BACKEND_PORT:-5000}
FRONTEND_PORT=${FRONTEND_PORT:-3000}
BACKEND_HOST=${BACKEND_HOST:-0.0.0.0}
FRONTEND_HOST=${FRONTEND_HOST:-localhost}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=================================================="
echo "CQA Chat System - Launch Script"
echo "==================================================${NC}"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created!${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed!${NC}"
fi

# Check if data file exists
if [ ! -f "data/cqa.csv" ]; then
    echo -e "${YELLOW}Data file not found. Generating dataset...${NC}"
    python src/make_data.py --out data/cqa.csv --n_per_cat 200
    echo -e "${GREEN}Dataset generated!${NC}"
fi

# Create log files
BACKEND_LOG="/tmp/api_server.log"
FRONTEND_LOG="/tmp/frontend_server.log"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down servers...${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit
}

trap cleanup SIGINT SIGTERM

# Start backend server with prefixed output
echo -e "${BLUE}Starting backend API server...${NC}"
python run_api.py 2>&1 | while IFS= read -r line; do
    echo -e "${BLUE}[BACKEND]${NC} $line" | tee -a "$BACKEND_LOG"
done &
BACKEND_PID=$!

# Wait for backend to start
echo -e "${BLUE}Waiting for backend to initialize...${NC}"
sleep 5

# Check if backend is running
BACKEND_URL="http://${BACKEND_HOST}:${BACKEND_PORT}"
if [ "$BACKEND_HOST" = "0.0.0.0" ]; then
    BACKEND_URL="http://localhost:${BACKEND_PORT}"
fi

if ! curl -s ${BACKEND_URL}/health > /dev/null; then
    echo -e "${YELLOW}Backend is still starting up. This may take a minute...${NC}"
    sleep 10
fi

# Check if frontend directory exists
if [ ! -d "frontend" ]; then
    echo -e "${YELLOW}Frontend directory not found. Please set up the React frontend first.${NC}"
    echo -e "${YELLOW}Run: cd frontend && npm install && npm run dev${NC}"
    echo ""
    DISPLAY_HOST=${BACKEND_HOST}
    if [ "$BACKEND_HOST" = "0.0.0.0" ]; then
        DISPLAY_HOST="localhost"
    fi
    echo -e "${GREEN}Backend is running at: http://${DISPLAY_HOST}:${BACKEND_PORT}${NC}"
    echo -e "${GREEN}API docs at: http://${DISPLAY_HOST}:${BACKEND_PORT}/docs${NC}"
    echo ""
    echo -e "${BLUE}Backend logs (Press Ctrl+C to stop):${NC}"
    echo -e "${BLUE}==================================================${NC}"
    # Wait for backend process (logs are already being displayed)
    wait $BACKEND_PID
else
    # Start frontend server with output to both file and terminal
    echo -e "${BLUE}Starting frontend development server...${NC}"
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}Installing frontend dependencies...${NC}"
        npm install
    fi
    
    # Link VITE_BACKEND_PORT to BACKEND_PORT for consistency
    export VITE_BACKEND_PORT=${BACKEND_PORT}
    export VITE_BACKEND_HOST=${BACKEND_HOST}
    export VITE_FRONTEND_PORT=${FRONTEND_PORT}
    
    npm run dev 2>&1 | while IFS= read -r line; do
        echo -e "${GREEN}[FRONTEND]${NC} $line" | tee -a "$FRONTEND_LOG"
    done &
    FRONTEND_PID=$!
    cd ..
    
    echo ""
    echo -e "${GREEN}=================================================="
    echo "✅ Both servers are starting!"
    echo "==================================================${NC}"
    echo ""
    # Display URLs (use localhost for display even if host is 0.0.0.0)
    DISPLAY_HOST=${BACKEND_HOST}
    if [ "$BACKEND_HOST" = "0.0.0.0" ]; then
        DISPLAY_HOST="localhost"
    fi
    
    echo -e "${GREEN}Backend API:${NC}  http://${DISPLAY_HOST}:${BACKEND_PORT}"
    echo -e "${GREEN}API Docs:${NC}      http://${DISPLAY_HOST}:${BACKEND_PORT}/docs"
    echo -e "${GREEN}Frontend:${NC}     http://${FRONTEND_HOST}:${FRONTEND_PORT}"
    echo ""
    echo -e "${YELLOW}Press Ctrl+C to stop both servers${NC}"
    echo ""
    echo -e "${BLUE}Note: You can override ports using environment variables:${NC}"
    echo -e "${BLUE}  BACKEND_PORT=5000 FRONTEND_PORT=3000 ./launch.sh${NC}"
    echo ""
    echo -e "${BLUE}=================================================="
    echo "Live Logs (Backend & Frontend):"
    echo "==================================================${NC}"
    echo ""
    
    # Wait for both processes (logs are already being displayed via the pipes above)
    wait $BACKEND_PID $FRONTEND_PID
fi

