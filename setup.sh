#!/bin/bash

echo "==============================================="
echo "   SquintyPics Setup Script"
echo "==============================================="
echo

echo "[1/4] Creating virtual environment..."
python3 -m venv .venv
if [ $? -ne 0 ]; then
    echo "Error creating virtual environment"
    exit 1
fi

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

echo "[3/4] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing dependencies"
    exit 1
fi

echo "[4/4] Setup complete!"
echo
echo "To run SquintyPics:"
echo "   1. Open terminal in this folder" 
echo "   2. Run: source .venv/bin/activate"
echo "   3. Run: streamlit run app.py"
echo
echo "The app will open at http://localhost:8501"
echo