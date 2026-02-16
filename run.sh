#!/bin/bash

# mimi - semantic meme search launcher

echo "🎨 Starting mimi - semantic meme search"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi

# Start the server
echo ""
echo "🚀 Starting backend server..."
echo "Frontend will be available at: http://localhost:8000"
echo ""
cd backend
python main.py
