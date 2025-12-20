#!/bin/bash
# clean_install_api.sh - Clean install of API dependencies

echo "================================================"
echo "🧹 Cleaning old dependencies..."
echo "================================================"

# Uninstall the conflicting package
echo "Removing old googletrans package..."
pip uninstall -y googletrans

echo ""
echo "================================================"
echo "📦 Installing fresh dependencies..."
echo "================================================"

# Install dependencies
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

