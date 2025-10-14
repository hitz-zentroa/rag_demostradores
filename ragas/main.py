"""
RAG Evaluation Script

This script evaluates Retrieval-Augmented Generation (RAG) systems using various metrics.
It loads a dataset containing user inputs, responses, and retrieved contexts, then performs
evaluation using a RAGEvaluator instance.

Requirements:
    - polars: For efficient dataframe operations
    - torch: For GPU memory management
    - yaml: For configuration file parsing
    - rag_evaluator: Custom module containing RAGEvaluator class
"""

import sys
import os
import asyncio
import yaml
import time
import argparse
import gc
import logging
import torch
import polars as pl

# Add current directory to Python path for local imports
sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/")
from rag_evaluator import RAGEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main(
    dataset: pl.DataFrame = None,
    metrics=None,
    evaluator: RAGEvaluator = None,
    output_path: str = ""
):
    """
    Main evaluation function that processes the dataset and generates evaluation metrics.
    
    Args:
        dataset (pl.DataFrame): Polars DataFrame containing evaluation data with columns:
            - user_inputs: User queries/questions
            - responses: System-generated responses
            - retrieved_contexts: Retrieved document contexts
            - references: Ground truth references (optional)
        metrics (list): List of evaluation metrics to compute
        evaluator (RAGEvaluator): Instance of RAGEvaluator for performing evaluation
        output_path (str): Path where evaluation results will be saved
    
    Returns:
        None: Results are saved to the specified output path
    
    Raises:
        SystemExit: If critical errors occur during data loading or evaluation
    """
    
    await evaluator.set_prompts()
    
    # Load user inputs from dataset
    try:
        user_inputs = dataset['user_inputs'].to_list()
    except Exception as e:
        logger.error(f"Error loading user_inputs: {e}")
        sys.exit(1)
    
    # Load system responses from dataset
    try:
        responses = dataset['responses'].to_list()
    except Exception as e:
        logger.error(f"Error loading responses: {e}")
        sys.exit(1)
    
    # Load and parse retrieved contexts
    try:
        retrieved_contexts = dataset['retrieved_contexts'].to_list()
    except Exception as e:
        logger.error(f"Error loading retrieved_contexts: {e}")
        sys.exit(1)
    
    # Load reference answers (optional)
    try:
        references = dataset['references'].to_list()
    except Exception as e:
        logger.warning(f"Error loading references: {e}")
        references = None
    
    # Perform evaluation
    df = await evaluator.evaluate(
        user_inputs=user_inputs,
        responses=responses,
        retrieved_contexts=retrieved_contexts,
        references=references,
        metrics=metrics,
        show_progress=True
    )
    
    # Save evaluation results to CSV
    try:
        df.sort("id")   
        df.write_ndjson(output_path)
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Script with configurable YAML path')
    parser.add_argument(
        '--config',
        type=str,
        default=f"{os.path.dirname(os.path.realpath(__file__))}/config.yaml",
        help='Path to the configuration YAML file'
    )
    args = parser.parse_args()
    
    # Load configuration from YAML file
    config = yaml.safe_load(open(args.config, "r"))
            
    # Initialize RAG evaluator with specified model
    try:
        evaluator = RAGEvaluator(
            id_model_llm=os.path.join(config['root-models'], config['MODEL_LLM_DIR']), 
            id_model_emb=os.path.join(config['root-models'], config['MODEL_EMB_DIR'])
        )
    except Exception as e:
        logger.error(f"Error initializing RAGEvaluator with models: {e}")
        sys.exit(1)
    
    # Load dataset with explicit schema
    try:
        source = os.path.join(config['root-dataset'], config['path-dataset'])
        dataset = pl.read_ndjson(
            source = source
        )
    except Exception as e:
        logger.error(f"Error loading dataset '{source}': {e}")
        logger.error(
            f"Please maintain dataset structure as: "
            f"user_inputs (String Utf8), responses (String Utf8), "
            f"retrieved_contexts (List of String Utf8), references (String Utf8)"
        )
        sys.exit(1)
        
    # Load evaluation metrics from configuration
    try:
        metrics = config['ragas']['metrics']
    except Exception as e:
        logger.error(f"Error loading metrics from config: {e}")
        sys.exit(1)
        
    # Construct output file path
    try:
        output_dir = os.path.join(config['root-output'])
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        output_path = os.path.join(
            output_dir, 
            "eval_" + config['path-dataset'] 
        )
        logger.info(f"Output path: {output_path}")
    except Exception as e:
        logger.error(f"Error constructing output path: {e}")
        sys.exit(1)
        
    # Execute evaluation and measure execution time
    start = time.time()
    asyncio.run(main(
        dataset=dataset,
        metrics=metrics,
        evaluator=evaluator,
        output_path=output_path
    ))
    logger.info(f"Execution time: {time.time() - start:.2f} seconds")
    
    # Clean up resources
    del evaluator
    gc.collect()
    torch.cuda.empty_cache()