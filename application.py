import os,sys
import traceback
import logging
import base64
import re
import ast

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from io import StringIO, BytesIO
from contextlib import contextmanager

from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from config import Config

import pickle
import json

import webbrowser
from threading import Timer

application = Flask(__name__)

socketio = SocketIO(application, cors_allowed_origins="*", manage_session=False,)

# Set logging level to WARNING
logging.getLogger('werkzeug').setLevel(logging.WARNING)

import scanpy as sc
sc.settings.verbosity = 0

from utils import init_sc,Talk2GPT,execute_and_fix_code

description='single cell RNA seq of ~3k PBMCs'
key_obs_columns = ['louvain']

with open('settings.json', 'r') as file:
    settings = json.load(file)


session={}
session['adata'] = sc.read(settings["adata_path"])
session['chat_history'] = []
session['extra_variables'] = {}

init_thread = init_sc(session['adata'],settings["project_description"],settings["obs_columns"])

@contextmanager
def capture_output():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout

@application.route('/')
def index():
    session['adata'] = sc.read(settings["adata_path"])
    session['chat_history'] = []
    session['extra_variables'] = {}
    return render_template('index.html',settings=settings)

@application.route('/reset_session', methods=['POST'])
def reset_session():
    # print('reset!')
    session['adata'] = sc.read(settings["adata_path"])
    session['chat_history'] = []
    session['extra_variables'] = {}
    return jsonify(success=True)

@application.route('/save_settings', methods=['POST'])
def save_settings():
    global settings
    settings["api_key"] = request.form.get("api-key")
    settings["adata_path"] = request.form.get("adata-path")
    settings["project_description"] = request.form.get("project-description")
    settings["obs_columns"] = [x.strip() for x in request.form.get("obs-columns").split(",")] 
    # print(settings)
    ### reset session
    session['adata'] = sc.read(settings["adata_path"])
    session['chat_history'] = []
    session['extra_variables'] = {}
    init_thread = init_sc(session['adata'],settings["project_description"],settings["obs_columns"])
    # save settings
    with open('settings.json', 'w') as fp:
        json.dump(settings, fp, indent=4)
    
    return jsonify({"status": "success"})

def log_chat(data):
    log_file = './logs/response.log'
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write(f"{timestamp} - {data}\n")

@socketio.on('chat_with_gpt')
def chat_with_gpt(data):

    Talk2GPT(session,data,init_thread)

@application.route('/info_message', methods=['POST'])
def info_message():
    message = request.form['message']
    print('------------\n'+message)
    socketio.emit("info_message", {"message": message})
    return jsonify(message=message)


@application.route('/execute', methods=['POST'])
def execute_code():
    code = request.form['code']
    prompt = request.form['prompt']
    prompt_history = json.loads(request.form['prompt_history'])
    code_history = json.loads(request.form['code_history'])
    output_history = json.loads(request.form['output_history'])

    complete_history = []
    for prompt_, code_, output_ in zip(prompt_history, code_history, output_history):
        # print(prompt_,output_)
        user_message = {"role": "user", "content": prompt_}
        assistant_message = {"role": "assistant", "content": '```python\n' + code_ + '```'}
        output_message = {"role": "user", "content": '[OUTPUT-INFO] The above code printed:\n'+output_}
        complete_history.append(user_message)
        complete_history.append(assistant_message)
        if len(output_)>0: complete_history.append(output_message)

    session['chat_history'] = complete_history

    fixed_code, output_str, plot_base64_list = execute_and_fix_code(code, prompt,session,init_thread,socketio)

    output_str = output_str[:1000] ## restrict output to 1000 characters

    log_chat(json.dumps([prompt,fixed_code,output_str]))
    
    return jsonify(fixed_code=fixed_code, output=output_str, plot_base64_list=plot_base64_list)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000/")

if __name__ == '__main__':
    Timer(1, open_browser).start()
    socketio.run(application,allow_unsafe_werkzeug=True)


