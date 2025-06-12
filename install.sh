#!/bin/bash

# Create virtual environment
python3.12 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Initialize the database
python init_auth.py

# Set permissions
chmod +x app.py

echo "Installation complete! Run 'source venv/bin/activate' to activate the virtual environment."
echo "Then run 'python app.py' to start the application."

