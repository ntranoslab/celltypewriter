#!/bin/bash

TERMINAL_PID=$(osascript -e 'tell application "Terminal" to do script "clear && cd '$PWD' && source venv/bin/activate && python application.py && deactivate && echo \"\nPress [Cmd + W] or [Cmd + Q] to close the Terminal window.\" && bash"' -e 'tell application "Terminal" to set custom title of window 1 to "CellTypeWriter"')
osascript -e 'tell application "Terminal" to activate'

while pgrep -q -s "$TERMINAL_PID"; do
    sleep 1
done
