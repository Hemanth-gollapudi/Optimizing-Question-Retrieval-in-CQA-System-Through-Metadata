# Launch Guide

## Quick Start (Recommended)

Use the launch script to start both backend and frontend:

```bash
./launch.sh
```

This will:
1. Check/create virtual environment
2. Install dependencies if needed
3. Generate dataset if missing
4. Start backend API server
5. Start frontend development server

## Manual Launch

### Backend Only

```bash
# Activate virtual environment
source venv/bin/activate

# Start API server
python run_api.py
```

Backend will be available at: `http://localhost:5000`

### Frontend Only

```bash
cd frontend
npm install  # First time only
npm run dev
```

Frontend will be available at: `http://localhost:3000`

## What's New

### React Frontend
- ✅ Converted to modular React components
- ✅ Fixed scrolling issue - page is now fully scrollable
- ✅ Better code organization
- ✅ Easier to extend and maintain

### Improved Launch
- ✅ Single command to start everything
- ✅ Automatic dependency checking
- ✅ Better error handling

## Troubleshooting

### Backend won't start
- Check if port 5000 is already in use
- Make sure virtual environment is activated
- Verify `data/cqa.csv` exists

### Frontend won't start
- Run `npm install` in the frontend directory
- Check if port 3000 is available
- Make sure backend is running first

### Port conflicts
- Backend: Change port in `run_api.py` (default: 5000)
- Frontend: Change port in `vite.config.js` (default: 3000)

