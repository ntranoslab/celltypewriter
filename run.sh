#!/bin/bash

TERMINAL_PID=$(osascript -e 'tell application "Terminal" to do script "clear && cd '$PWD' && source venv/bin/activate && python application.py && deactivate && echo \"\nPress [Cmd + W] or [Cmd + Q] to close the Terminal window.\" && bash"' -e 'tell application "Terminal" to set custom title of window 1 to "CellTypeWriter"')
osascript -e 'tell application "Terminal" to activate'

while pgrep -q -s "$TERMINAL_PID"; do
    sleep 1
done

# TERMINAL_PID=$(osascript -e 'tell application "Terminal" to do script "cd '$PWD' && source venv/bin/activate && python application.py && deactivate && echo \"\nPress [Cmd + W] or [Cmd + Q] to close the Terminal window.\" && bash"')
# osascript -e 'tell application "Terminal" to activate'

# while pgrep -q -s "$TERMINAL_PID"; do
#     sleep 1
# done


# export LC_ALL=en_US.UTF-8
# export LANG=en_US.UTF-8

# source venv/bin/activate
# python3 application.py
