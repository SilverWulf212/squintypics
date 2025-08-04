@echo off
echo ===============================================
echo   SquintyPics Setup Script
echo ===============================================
echo.

echo [1/4] Creating virtual environment...
python -m venv .venv
if %errorlevel% neq 0 (
    echo Error creating virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

echo [3/4] Installing dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error installing dependencies
    pause
    exit /b 1
)

echo [4/4] Setup complete!
echo.
echo To run SquintyPics:
echo   1. Open terminal in this folder
echo   2. Run: .venv\Scripts\activate
echo   3. Run: streamlit run app.py
echo.
echo The app will open at http://localhost:8501
echo.
pause