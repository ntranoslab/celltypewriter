import os,sys
import traceback
import logging
import base64
import re

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import scanpy as sc
sc.settings.verbosity = 0

from io import StringIO, BytesIO
from contextlib import contextmanager

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import requests

import pickle
import json

import openai

with open('settings.json', 'r') as file:
    settings = json.load(file)

openai.api_key =  settings['api_key']


def init_sc(adata,description,key_obs_columns):
  sys_prompt=[{"role": "system", "content": \
"""You are a helpful programming assistant. Your goal is to understand and translate quick notes to python code.
You *never* respond with plain text.
Your responses only provide code that can be executed."""},]
  initial_prompt = \
"""You are a helping me explore my single-cell RNA seq dataset.
I am going to describe what I want to plot and you will respond with code that generates the plot.
You *never* respond with plain text. Your responses only provide a ``` code block ``` that can be executed independently.\n
To print something always use the print() function. When something is printed in the output I will report that back with an [OUTPUT-INFO] message so that you can use it in followup requests. Followup requests are more likely to refer to the most recent code block. 

The variable adata is loaded and it is the name of the AnnData object we'll explore in this session:
"""
  cols_info = ''
  cols='\nadata.obs['
  for k in key_obs_columns:
    if k in adata.obs.columns:
        cols_info+=k+', '
        cols+='\''+k+'\']'+':\n'+str(adata.obs[k].values)+'\n'
  if cols == '\nadata.obs[' : cols=''

  adata_info='\n'+str(adata)+'\n'
  
  adata_info+='\nadata.X contains the processed gene expression data, for only highly variable genes, datatype: '+str(type(adata.X))+', shape: '+str(adata.X.shape)+'\n'
  adata_info+='adata.raw.X contains the raw gene expression data for all genes, datatype: '+str(type(adata.raw.X))+', shape: '+str(adata.raw.X.shape)+'\n'
  adata_info+='Use adata.raw.X to get gene expression values, unless specified otherwise. Keep in mind that adata.raw.X datatype is '+str(type(adata.raw.X))+'\n'

  initial_prompt+='\n---\n'+'Project Description: '+description+'.\n\nPlease review and remember the structure of my AnnData object. \nThe variable is called adata and contains the following information:\n\n'+adata_info+cols
  initial_prompt+='\nNever change the original adata and minimize usage of scanpy.tl functions. If any computation is needed, the original adata should be backed up (e.g., adata_backup = adata.copy()) so we can revert to it later. Plot requests are typically just scanpy.pl functions. Using seaborn is preferred for dataframes.'
  initial_prompt+="""
I have already imported the following modules: 
  - scanpy as sc 
  - pandas as pd
  - numpy as np 
  - seaborn as sns, 
  - matplotlib.pyplot as plt"""
  
  if not os.path.exists('./logs'):
    os.mkdir('./logs')

  with open('./logs/init.log','w') as f:
    f.write(initial_prompt)

  init_thread=sys_prompt + [{"role": "user", "content": \
                   initial_prompt},]
  init_thread+=[{"role": "assistant", "content": \
                   "```python\nprint(I understand. I will translate your commands to code.)\n```"},]
  return init_thread



def Talk2GPT(session,data,init_thread):
        message = data['message']
        prompt_history = data['prompt_history']
        code_history = data['code_history']
        output_history = data['output_history']
        
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


        message+=' -- Please respond with ```python code``` only no text'
        
        current_thread = init_thread + session['chat_history'] + [{"role": "user", "content": message},]

        response = openai.ChatCompletion.create(
            model="gpt-4", #"gpt-3.5-turbo",
            messages=current_thread,
            max_tokens=500,
            temperature=0,
            stream=True
        )
        # print(response)
        code_buffer = ""
        emit_flag = False

        for chunk in response:
            if 'content' in chunk['choices'][0]['delta'].keys():
                reply = chunk['choices'][0]['delta']['content']

                # Accumulate the text in the code buffer
                code_buffer += reply
                # print(code_buffer)

                # Check if a complete Python code block is available
                code_match = re.search(r'```python\n(.*?)', code_buffer, re.DOTALL)

                if code_match:
                    # Emit the code inside the matched block
                    if emit_flag:
                        # reply = reply.strip()
                        reply = reply.rstrip('```')
                        emit('gpt_reply', reply)

                    emit_flag = True

                    # Check for the next Python code block
                    block_done = re.search(r'```python(.*?)```', code_buffer, re.DOTALL)
                    if block_done:
                        # Remove the matched block from the code buffer
                        code_buffer = code_buffer.replace(block_done.group(0), '', 1)

@contextmanager
def capture_output():
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        yield sys.stdout
    finally:
        sys.stdout = old_stdout


def execute_and_fix_code(code,prompt,session,init_thread,socketio):
    fixed_code = code
    output_str = ''
    plot_base64_list = []
    N_ITER = 4 
    current_error = ''

    fix_thread = init_thread+session['chat_history']+[{"role": "user", "content": prompt},{"role": "assistant", "content": '```python\n'+fixed_code+'```'},]
    
    for k in range(N_ITER):

        # Capture the output using the custom context manager
        with capture_output() as output:
            
            exec_namespace = {**{'sns': sns, 'plt': plt,'sc':sc,'np':np,'pd':pd,'adata':session['adata']},**session['extra_variables']}

            try:                
                # Execute the user's code
                exec(fixed_code, exec_namespace) # Add imports to the execution context
            except Exception as e:
                output.write('Error: ' + str(e))
                current_error = str(e)

        session['adata'] = exec_namespace['adata']
        for variable in set(exec_namespace.keys()).difference(['sns','plt','sc','pd','np','adata',]):
            if not variable.startswith('__'):
                try:
                    pickle.dumps(exec_namespace[variable])
                    session['extra_variables'][variable] = exec_namespace[variable]
                except (pickle.PicklingError, TypeError):
                    # print('Session Error: variable',variable)
                    pass
        
        # Get the output and check for errors
        output_str = output.getvalue()

        # save original code
        if k==0: bad_code=fixed_code

        if (not output_str.startswith('Error:')):# or (len(prompt) == 0):
            break

        if k==0:
            print(f"\n\033[32m[info]\033[0m Something is not right. I'll try to fix it. Please wait...\n\033[32m[info]\033[0m fixing error #{k+1}: \033[90m{current_error}\033[0m")
            info_message = f"""
            <span style="color: #32CD32;">[info]</span> Something is not right. I'll try to fix it. Please wait...
            <br><span style="color: #32CD32;">[info]</span> fixing error #{k+1}: <span style="color: #909090;">{current_error}</span>
            """
            errors_ = current_error+', '
        else:
            print(f"\033[32m[info]\033[0m fixing error #{k+1}: \033[90m{current_error}\033[0m")
            info_message = f"""
            <span style="color: #32CD32;">[info]</span> fixing error #{k+1}: <span style="color: #909090;">{current_error}</span>
            """
            errors_ += current_error+', '

        socketio.emit("info_message", {"message": info_message})

        # print('Error: ' +current_error))
        fix_thread += [  {"role": "user", "content": 'I got this error: '+current_error+'. Please fix this. Note: fixed code should be self-contained.' }]
        # print(fix_thread)
        response = openai.ChatCompletion.create(
            model="gpt-4", #"gpt-3.5-turbo",
            messages=fix_thread,
            max_tokens=500,
            temperature=0,
        )
        # print('INFO:','fixed code -- retrying')
        plt.close('all')

        code_pattern = r"```python\n(.*?)```"
        matches = re.findall(code_pattern, response['choices'][0]['message']['content'], re.DOTALL)

        if len(matches)>0: 
            fixed_code = matches[0]
        else:
            break

        fix_thread +=[{"role": "assistant", "content": '```python\n'+fixed_code+'```'}]
        # print(fixed_code)
        # print('---')

    # Save all the figures as base64-encoded images
    for i, fig in enumerate(plt.get_fignums()):
        buf = BytesIO()
        plt.figure(fig)
        plt.tight_layout()  # Adjust the figure size to fit all elements
        plt.savefig(buf, format="png", bbox_inches='tight')  # Save the figure with the adjusted size
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('ascii')
        plot_base64_list.append(plot_base64)

    # Clear all the figures to avoid overlapping plots
    plt.close('all')

    if not (bad_code == fixed_code):

        ask = f"""Tried to execute this code:
        {bad_code} ///
        and got the following errors:
        {errors_} ///
        Your fixed version:
        {fixed_code} ///
        --- Write a very short comment on what you changed.
        """

        response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[  {"role": "user", "content": ask }],
                max_tokens=100,
                temperature=0,
            )

        fixed_code = '#'+f"""Note: {response['choices'][0]['message']['content']}""" +'\n\n'+ fixed_code

    return fixed_code, output_str, plot_base64_list

