#!/bin/bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
echo "Setup complete. You can now run the application with ./run.sh"
