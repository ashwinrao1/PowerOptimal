@echo off
REM Quick start script to run the Streamlit dashboard (Windows)

echo Starting Data Center Energy Optimization Dashboard...
echo.

REM Check if streamlit is installed
where streamlit >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Streamlit is not installed
    echo Please install requirements: pip install -r requirements.txt
    exit /b 1
)

REM Run the dashboard
streamlit run dashboard\app.py
