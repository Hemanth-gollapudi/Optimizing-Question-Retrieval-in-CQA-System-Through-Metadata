#!/bin/bash
# install_api.sh - Install FastAPI backend dependencies

echo "================================================"
echo "🚀 Installing ITRLM+RAG API Dependencies"
echo "================================================"

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   Consider activating your environment first:"
    echo "   source venv/bin/activate  # or your environment"
    echo ""
fi

# Install dependencies
echo "📦 Installing dependencies from requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Installation complete!"
    echo ""
    echo "================================================"
    echo "🎉 Next Steps:"
    echo "================================================"
    echo "1. Start the server:"
    echo "   python run_server.py"
    echo ""
    echo "2. Or with auto-reload (development):"
    echo "   python run_server.py --reload"
    echo ""
    echo "3. Access API documentation:"
    echo "   http://localhost:8000/docs"
    echo ""
    echo "4. Test the API:"
    echo "   python test_api.py"
    echo "================================================"
else
    echo ""
    echo "❌ Installation failed!"
    echo "   Please check the error messages above"
    exit 1
fi

