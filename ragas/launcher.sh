#!/bin/bash

CONFIG_PATH="./config.yaml" # Modificar ruta del archivo config.yaml si es necesario

# Ejecución del script
echo "Iniciando evaluación de RAG con Ragas..."
srun python main.py --config $CONFIG_PATH
echo "Evaluación completada."
