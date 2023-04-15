# CellTypeWriter

CellTypeWriter is a user-friendly web application for exploring and analyzing single-cell RNA-seq data using an interactive chat interface with an AI-based language model (GPT-4). It offers dynamic code execution and output visualization, along with adjustable settings for data input and analysis.

## Functionality

The CellTypeWriter web app provides an interactive interface for users to explore and analyze single-cell RNA-seq data with the help of the GPT-4 AI model. The main features of the web app include:

- **Interactive Chat Interface**: The app features a chat interface where users can ask questions or request analysis tasks to be performed by the GPT-4 AI model. The model then responds with appropriate Python code, which is added to the code editor.
- **Code Editor**: The app includes a code editor, based on the CodeMirror library, for writing and editing Python code. Users can execute the code by pressing Shift + Enter or by clicking the "Execute Code" button. The code editor allows users to view and modify the code generated by the GPT-4 AI model.
- **Output Display and Visualization**: The output of the executed code is displayed in the "Output" section, along with any plots generated during the code execution. Plots are rendered as base64-encoded images within the "Plots" section.
- **Prompt History**: The app maintains a history of user prompts and the corresponding AI-generated code, output, and plots. Users can click on a prompt in the history to restore the code editor, output, and plot states for that prompt.
- **Settings**: Users can access the settings modal by clicking the "Settings" button. The settings modal allows users to provide their API key, the path to their AnnData file, a project description, and a list of observation columns for analysis. The app saves these settings and uses them for further interactions.
- **Reset Session**: Users can reset the session, which clears the prompt history and resets the server-side session. This action can be performed by clicking the "Reset History" button.

## Installation

### MacOS

1. Download the repository and open the `Setup.app` to install the application.
2. Run the `CellTypeWriter.app` to start the application.

### command line

1. Clone the repository and change to the `celltypewriter` directory:
```
git clone https://github.com/ntranoslab/celltypewriter.git
cd celltypewriter
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

## Requirements

To use the CellTypeWriter web app, users must have an OpenAI API key with access to the GPT-4 API. The app relies on the GPT-4 AI model to generate code snippets for single-cell RNA-seq data analysis.

### Obtaining an OpenAI API Key

To obtain an OpenAI API key, you will need to sign up for an account on the [OpenAI website](https://www.openai.com/). Once you have an account, you can request access to the GPT-4 API. After you have been granted access, you will find your API key in the [API keys section](https://platform.openai.com/signup) of your OpenAI account.

### Providing the API Key to the Web App

After obtaining the API key, you can provide it to the CellTypeWriter web app through the "Settings" modal. Click the "Settings" button on the web app's main page, and enter your API key in the "API Key" field. Save the settings, and the web app will use your API key to communicate with the GPT-4 API for all subsequent interactions.

## Disclaimer

Please note that the analysis and visualization code provided by the CellTypeWriter web app is generated by GPT-4, an advanced AI language model developed by OpenAI. While GPT-4 can often generate useful and accurate code, it is essential to exercise caution and perform due diligence when using this tool for research purposes. Some important points to consider are:

1. **Research Integrity**: Relying solely on generated code without understanding the underlying methods and assumptions can lead to misleading or incorrect conclusions. It is crucial to develop a strong foundation in the methods used in your research and understand the context in which they should be applied.

2. **Data Examination**: Before using the generated code, carefully examine your data to ensure its quality and suitability for the intended analysis. It is essential to understand the limitations and potential biases in your data, as these can impact the validity of your results.

3. **Code Validation**: Always review the generated code to ensure its correctness and relevance to your specific use case. This includes checking for accuracy, completeness, and optimization. Be prepared to modify the code as needed to meet your research objectives.

4. **Continuous Learning**: Stay up-to-date with the latest advancements in your field and be aware of any updates to the tools and libraries you are using. Regularly re-evaluate the generated code to ensure it aligns with the current best practices and standards in your domain.

By using the CellTypeWriter web app, you acknowledge these considerations and agree to use the generated code responsibly. Thoroughly review and validate the generated code before incorporating it into your analysis.




