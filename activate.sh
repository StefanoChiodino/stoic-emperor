#!/bin/bash
# Quick activation helper
# Usage: source activate.sh

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found. Run: source setup.sh"
    return 1 2>/dev/null || exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
    echo "✅ Environment variables loaded"
fi

echo ""
echo "Ready to use! Run: python src/run_pipeline.py"
