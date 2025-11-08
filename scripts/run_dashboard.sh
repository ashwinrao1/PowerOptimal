#!/bin/bash
# Quick start script to run the Streamlit dashboard

echo "Starting Data Center Energy Optimization Dashboard..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null
then
    echo "Error: Streamlit is not installed"
    echo "Please install requirements: pip install -r requirements.txt"
    exit 1
fi

# Run the dashboard
streamlit run dashboard/app.py
