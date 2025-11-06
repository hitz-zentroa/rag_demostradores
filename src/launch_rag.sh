#! /bin/bash

# Activate the Python environment
# source path/to/your/venv/bin/activate

CONFIG_FILE="./config_base.json"
INDICES_CONFIG="./es_indices.json"

# Run the Gradio interface

python3 rag_interface.py $CONFIG_FILE $INDICES_CONFIG