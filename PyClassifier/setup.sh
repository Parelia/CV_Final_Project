#! /bin/bash

# create venv
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    source .venv/bin/activate
else
    source .venv/bin/activate
fi

# install dependencies
pip install -r requirements.txt