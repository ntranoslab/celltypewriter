# CellTypeWriter

CellTypeWriter is a user-friendly web application for exploring and analyzing single-cell RNA-seq data using an interactive chat interface with an AI-based language model (GPT-4). It offers dynamic code execution and output visualization, along with adjustable settings for data input and analysis.

## Installation

### MacOS

1. Download the repository and open the `Setup.app` to install the application.
2. Run the `CellTypeWriter.app` to start the application.

### command line

1. Clone the repository:
```
git clone https://github.com/ntranoslab/celltypewriter.git
```
2. Set up the environment:
```
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
3. Start the application:
```
python application.py
```

## Usage

1. Run the `CellTypeWriter.app` (MacOS) or execute `python application.py` (command line).
2. A browser window will automatically launch with the app at `http://127.0.0.1:5000/`.
3. Interact with the application by typing questions or commands in the chat interface.
