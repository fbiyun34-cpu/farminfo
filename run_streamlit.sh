#!/bin/bash

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "python3 could not be found. Please install Python 3."
    exit 1
fi

echo "Installing dependencies..."
pip3 install -r output/requirements.txt

echo "Starting Streamlit Dashboard..."
streamlit run output/dashboard.py
