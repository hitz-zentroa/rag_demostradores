"""
RAG Evaluator Module

This module provides a comprehensive evaluation framework for Retrieval-Augmented Generation (RAG)
systems. It supports multiple evaluation metrics including context precision, context recall,
faithfulness, response relevancy, and context relevance.

The RAGEvaluator class wraps various metrics from the RAGAS framework and provides async
evaluation capabilities for efficient batch processing.

Typical usage:
    evaluator = RAGEvaluator(model_id="gpt2")
    results = await evaluator.evaluate(
        user_inputs=["What is AI?"],
        responses=["AI is..."],
        retrieved_contexts=[["context1", "context2"]],
        metrics=["FAITHFULNESS", "RESPONSE_RELEVANCY"]
    )
"""

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFacePipeline
from transformers import BitsAndBytesConfig
from ragas import SingleTurnSample
from ragas.metrics import ResponseRelevancy
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextPrecisionWithReference,
    ContextRelevance
)

import os, sys, yaml, time, asyncio
from enum import Enum, auto
from tqdm.asyncio import tqdm_asyncio
import polars as pl
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class MetricName(Enum):
    """
    Enumeration of available evaluation metrics for RAG systems.
    
    Attributes:
        CONTEXT_PRECISION_WOREF: Context precision without reference answers
        CONTEXT_PRECISION_WREF: Context precision with reference answers
        CONTEXT_RECALL: Measures how much of the reference is covered by retrieved context
        FAITHFULNESS: Measures factual consistency of response with retrieved context
        RESPONSE_RELEVANCY: Measures how relevant the response is to the user input
        CONTEXT_RELEVANCE: Measures how relevant the retrieved context is to the user input
    """
    CONTEXT_PRECISION_WOREF = auto()
    CONTEXT_PRECISION_WREF = auto()
    CONTEXT_RECALL = auto()
    FAITHFULNESS = auto()
    RESPONSE_RELEVANCY = auto()
    CONTEXT_RELEVANCE = auto()


class MissingParameterError(Exception):
    """
    Exception raised when required parameters are missing for a specific metric evaluation.
    """
    pass


class RAGEvaluator:
    """
    Comprehensive evaluator for RAG (Retrieval-Augmented Generation) systems.
    
    This class initializes and manages multiple evaluation metrics, providing both
    individual metric evaluation and batch evaluation capabilities with async support.
    
    Attributes:
        config (dict): Configuration loaded from YAML file
        logger (logging.Logger): Logger instance for this class
        embeddings: HuggingFace embeddings model
        llm: Language model for evaluation
        evaluator_embeddings: Wrapped embeddings for RAGAS
        evaluator_llm: Wrapped LLM for RAGAS
        scorers (dict): Dictionary mapping MetricName to metric scorer instances
    """
    
    def __init__(self, id_model_llm, id_model_emb):
        """
        Initialize the RAG evaluator with specified model.
        
        Args:
            id_model (str): Model identifier/path for the language model to use
        
        Raises:
            Exception: If model or embeddings loading fails
        """
        # Load configuration from YAML file
        self.config = yaml.safe_load(
            open(f"{os.path.dirname(os.path.realpath(__file__))}/config.yaml", "r")
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize embeddings and language models
        self.embeddings = self._get_embeddings(id_model_emb)
        self.llm = self._get_llm(id_model_llm)
        
        # Wrap models for RAGAS framework
        self.evaluator_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        self.evaluator_llm = LangchainLLMWrapper(self.llm)
        
        # Initialize all metric scorers
        self.scorers = {
            MetricName.CONTEXT_PRECISION_WOREF: LLMContextPrecisionWithoutReference(llm=self.evaluator_llm),
            MetricName.CONTEXT_PRECISION_WREF: LLMContextPrecisionWithReference(llm=self.evaluator_llm),
            MetricName.CONTEXT_RECALL: LLMContextRecall(llm=self.evaluator_llm),
            MetricName.FAITHFULNESS: Faithfulness(llm=self.evaluator_llm),
            MetricName.RESPONSE_RELEVANCY: ResponseRelevancy(llm=self.evaluator_llm, embeddings=self.evaluator_embeddings),
            MetricName.CONTEXT_RELEVANCE: ContextRelevance(llm=self.evaluator_llm)
        }
            
    async def set_prompts(self):
        """
        Configure all metric scorers to use Spanish language prompts.
        
        This method loads Spanish prompts from the configured path and applies them
        to all available metric scorers. Useful for evaluating Spanish language RAG systems.
        """
        
        prompts_path = os.path.join(
            self.config['root'], 
            self.config['ragas']['prompts']
        )
        
        # Create prompts directory if it doesn't exist
        if not os.path.exists(prompts_path):
            os.makedirs(prompts_path, exist_ok=True)
        
        # Set Spanish prompts for each metric
        # If prompts directory is empty, save default prompts in spanish
        prompt_list = [p.replace(".json", "") for p in os.listdir(prompts_path)]
        
        for metric_name, scorer in self.scorers.items():
            
            try:
                # !!!
                self.logger.info(f"Checking prompts for {scorer.name} in {self.config['ragas']['language']}")
                prompt_list = list(scorer.get_prompts().keys())
                possible_file_prompts = [f"{scorer.name}_{p}_{self.config['ragas']['language']}.json" for p in prompt_list]
                if any(prompt_file in os.listdir(prompts_path) for prompt_file in possible_file_prompts):
                    self.logger.info(f"Prompts for {scorer.name} already exist in {self.config['ragas']['language']}")
                else:
                    adapted_prompts = await scorer.adapt_prompts(language=self.config['ragas']['language'], llm=self.llm)
                    scorer.set_prompts(**adapted_prompts)
                    scorer.save_prompts(prompts_path)
            except Exception as e:
                self.logger.error(f"Error adapting prompts: {e}\n")
                self.logger.error(traceback.format_exc())
            
            try:
                scorer.set_prompts(
                    **scorer.load_prompts(
                        path=prompts_path,
                        language=self.config['ragas']['language']
                    )
                )
            except Exception as e:
                self.logger.error(f"Error setting adapted prompt: {e}\n")
                self.logger.error(traceback.format_exc())
                sys.exit(1)
    
    def _get_embeddings(self, id_model=""):
        """
        Load and initialize the embeddings model from HuggingFace.
        
        Args:
            id_model (str): Model identifier/path for the LLM

        Returns:
            HuggingFaceEmbeddings: Initialized embeddings model
        
        Raises:
            Exception: If embeddings model loading fails
        """
        
        try:
            model = HuggingFaceEmbeddings(model_name=id_model)
        except Exception as e:
            self.logger.error(
                f"Error loading embeddings model {id_model}: {e}"
            )
            raise e
        
        return model
    
    def _get_llm(self, id_model=""):
        """
        Load and initialize the language model from HuggingFace.
        
        Args:
            id_model (str): Model identifier/path for the LLM
        
        Returns:
            HuggingFacePipeline: Initialized language model pipeline
        
        Raises:
            Exception: If model loading fails
        """
        start = time.time()
        
        try:
            model = HuggingFacePipeline.from_model_id(
                model_id=id_model,
                task=self.config["llm"].get("task", "text-generation"),
                pipeline_kwargs={
                    "max_new_tokens": self.config["llm"]["pipeline_kwargs"].get("max_new_tokens", 2048),
                    "repetition_penalty": self.config["llm"]["pipeline_kwargs"].get("repetition_penalty", 1.03),
                    "return_full_text": self.config["llm"]["pipeline_kwargs"].get("return_full_text", False),
                    "device_map": self.config["llm"]["pipeline_kwargs"].get("device_map", "auto"),
                    "dtype": self.config["llm"]["pipeline_kwargs"].get("dtype", "auto")
                }
            )
        except Exception as e:
            self.logger.error(f"Error loading model {id_model}: {e}")
            raise e
        
        self.logger.info(
            f"Execution time - LLM load: {time.time() - start:.2f} seconds"
        )
        
        return model
    
    # METRICS
    
    async def _evaluate_context_precision_without_reference(
        self, 
        user_input, 
        response, 
        retrieved_contexts
    ):
        """
        Evaluate context precision without reference answers.
        
        Measures how precise the retrieved contexts are for generating the response.
        
        Args:
            user_input (str): User's query or question
            response (str): System-generated response
            retrieved_contexts (list[str]): List of retrieved context documents
        
        Returns:
            float: Context precision score
        
        Raises:
            MissingParameterError: If required parameters are None
        """
        if user_input is None or response is None or retrieved_contexts is None:
            raise MissingParameterError(
                "Context Precision without reference requires user_input, "
                "response and retrieved_contexts."
            )
        
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        
        return await self.scorers[MetricName.CONTEXT_PRECISION_WOREF].single_turn_ascore(sample)
    
    async def _evaluate_context_precision_with_reference(
        self, 
        user_input, 
        response, 
        retrieved_contexts, 
        reference
    ):
        """
        Evaluate context precision with reference answers.
        
        Measures how precise the retrieved contexts are compared to ground truth reference.
        
        Args:
            user_input (str): User's query or question
            response (str): System-generated response
            retrieved_contexts (list[str]): List of retrieved context documents
            reference (str): Ground truth reference answer
        
        Returns:
            float: Context precision score
        
        Raises:
            MissingParameterError: If required parameters are None
        """
        if user_input is None or response is None or retrieved_contexts is None:
            raise MissingParameterError(
                "Context Precision with reference requires user_input, "
                "response, retrieved_contexts and reference."
            )
        
        sample = SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
            reference=reference
        )
        
        return await self.scorers[MetricName.CONTEXT_PRECISION_WREF].single_turn_ascore(sample)
    
    async def _evaluate_context_recall(
        self, 
        user_input, 
        response, 
        retrieved_contexts, 
        reference
    ):
        """
        Evaluate context recall against reference answer.
        
        Measures how much of the reference answer is covered by the retrieved contexts.
        
        Args:
            user_input (str): User's query or question
            response (str): System-generated response
            retrieved_contexts (list[str]): List of retrieved context documents
            reference (str): Ground truth reference answer
        
        Returns:
            float: Context recall score
        
        Raises:
            MissingParameterError: If required parameters are None
        """
        if (user_input is None or response is None or 
            retrieved_contexts is None or reference is None):
            raise MissingParameterError(
                "Context Recall requires user_input, response, "
                "retrieved_contexts and reference."
            )
        
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            reference=reference,
        )
        
        return await self.scorers[MetricName.CONTEXT_RECALL].single_turn_ascore(sample)
    
    async def _evaluate_faithfulness(self, user_input, response, retrieved_contexts):
        """
        Evaluate faithfulness of the response to retrieved contexts.
        
        Measures whether the response is factually consistent with the retrieved contexts.
        
        Args:
            user_input (str): User's query or question
            response (str): System-generated response
            retrieved_contexts (list[str]): List of retrieved context documents
        
        Returns:
            float: Faithfulness score
        
        Raises:
            MissingParameterError: If required parameters are None
        """
        if user_input is None or response is None or retrieved_contexts is None:
            raise MissingParameterError(
                "Faithfulness requires user_input, response and retrieved_contexts."
            )
        
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        
        return await self.scorers[MetricName.FAITHFULNESS].single_turn_ascore(sample)
    
    async def _evaluate_response_relevancy(
        self, 
        user_input, 
        response, 
        retrieved_contexts
    ):
        """
        Evaluate relevancy of the response to the user input.
        
        Measures how relevant and appropriate the response is for the user's query.
        
        Args:
            user_input (str): User's query or question
            response (str): System-generated response
            retrieved_contexts (list[str]): List of retrieved context documents
        
        Returns:
            float: Response relevancy score
        
        Raises:
            MissingParameterError: If required parameters are None
        """
        if user_input is None or response is None or retrieved_contexts is None:
            raise MissingParameterError(
                "Response Relevancy requires user_input, response and retrieved_contexts."
            )
        
        sample = SingleTurnSample(
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
        
        return await self.scorers[MetricName.RESPONSE_RELEVANCY].single_turn_ascore(sample)
    
    async def _evaluate_context_relevance(self, user_input, retrieved_contexts):
        """
        Evaluate relevance of retrieved contexts to the user input.
        
        Measures how relevant the retrieved documents are to answering the user's query.
        
        Args:
            user_input (str): User's query or question
            retrieved_contexts (list[str]): List of retrieved context documents
        
        Returns:
            float: Context relevance score
        
        Raises:
            MissingParameterError: If required parameters are None
        """
        if user_input is None or retrieved_contexts is None:
            raise MissingParameterError(
                "Context Relevance requires user_input and retrieved_contexts."
            )
        
        sample = SingleTurnSample(
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
        )
        
        return await self.scorers[MetricName.CONTEXT_RELEVANCE].single_turn_ascore(sample)
    
    # EVALUATION
    
    async def _evaluate(
        self,
        idx=0,
        user_input=None,
        response=None,
        retrieved_contexts=None,
        reference=None,
        metrics=None,
    ):
        """
        Internal method to evaluate a single sample across multiple metrics.
        
        Args:
            idx (int): Sample index for tracking
            user_input (str): User's query or question
            response (str): System-generated response
            retrieved_contexts (list[str]): List of retrieved context documents
            reference (str, optional): Ground truth reference answer
            metrics (list[str], optional): List of metric names to evaluate. 
                If None, evaluates all available metrics.
        
        Returns:
            dict: Dictionary containing evaluation results for all requested metrics
        
        Notes:
            - Metrics requiring references will be skipped if reference is None
            - Errors in individual metrics are caught and stored as error messages
        """
        # Default to all metrics if none specified
        if metrics is None:
            metrics = [
                MetricName.CONTEXT_PRECISION_WOREF,
                MetricName.CONTEXT_PRECISION_WREF,
                MetricName.CONTEXT_RECALL,
                MetricName.FAITHFULNESS,
                MetricName.RESPONSE_RELEVANCY,
                MetricName.CONTEXT_RELEVANCE
            ]
        else:
            # Convert string metric names to enum
            metrics = [MetricName[m] for m in metrics]
        
        # Initialize result dictionary
        result = {
            "id": idx,
            "user_inputs": user_input,
            "responses": response,
            "retrieved_contexts": str(retrieved_contexts),
            "references": reference if reference else ""
        }
        
        # Evaluate each requested metric
        for metric in metrics:
            try:
                if metric == MetricName.CONTEXT_PRECISION_WOREF:
                    score = await self._evaluate_context_precision_without_reference(
                        user_input, response, retrieved_contexts
                    )
                elif metric == MetricName.CONTEXT_PRECISION_WREF and reference:
                    score = await self._evaluate_context_precision_with_reference(
                        user_input, response, retrieved_contexts, reference
                    )
                elif metric == MetricName.CONTEXT_RECALL and reference:
                    score = await self._evaluate_context_recall(
                        user_input, response, retrieved_contexts, reference
                    )
                elif metric == MetricName.FAITHFULNESS:
                    score = await self._evaluate_faithfulness(
                        user_input, response, retrieved_contexts
                    )
                elif metric == MetricName.RESPONSE_RELEVANCY:
                    score = await self._evaluate_response_relevancy(
                        user_input, response, retrieved_contexts
                    )
                elif metric == MetricName.CONTEXT_RELEVANCE:
                    score = await self._evaluate_context_relevance(
                        user_input, retrieved_contexts
                    )
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                result[metric.name] = score
            
            except Exception as e:
                result[metric.name] = f"Error: {str(e)}"
        
        return result
    
    async def evaluate(
        self,
        user_inputs,
        responses,
        retrieved_contexts,
        references=None,
        metrics=None,
        show_progress=True,
    ):
        """
        Evaluate a complete dataset of RAG system outputs across multiple metrics.
        
        This method processes multiple samples in parallel for efficient batch evaluation.
        
        Args:
            user_inputs (list[str]): List of user queries/questions
            responses (list[str]): List of system-generated responses
            retrieved_contexts (list[list[str]]): List of retrieved context lists for each query
            references (list[str], optional): List of ground truth reference answers
            metrics (list[str], optional): List of metric names to evaluate. 
                If None, evaluates all available metrics.
            show_progress (bool, optional): Whether to display progress bar. Defaults to True.
        
        Returns:
            pl.DataFrame: Polars DataFrame containing evaluation scores for all samples
        
        Raises:
            ValueError: If input lists have inconsistent lengths
        
        Example:
            >>> evaluator = RAGEvaluator("path_to_gpt2")
            >>> results = await evaluator.evaluate(
            ...     user_inputs=["What is AI?", "Explain ML"],
            ...     responses=["AI is...", "ML is..."],
            ...     retrieved_contexts=[["context1"], ["context2"]],
            ...     metrics=["FAITHFULNESS", "RESPONSE_RELEVANCY"]
            ... )
        """
        
        # Validate input lengths
        n = len(user_inputs)
        if not (len(responses) == len(retrieved_contexts) == n and (references is None or len(references) == n)):
            raise ValueError(
                "All input lists must have the same length."
            )
                
        # Prepare evaluation tasks for parallel execution
        tasks = []
        for i in range(n):
            tasks.append(
                self._evaluate(
                    idx=i,
                    user_input=user_inputs[i],
                    response=responses[i],
                    retrieved_contexts=retrieved_contexts[i],
                    reference=references[i] if references else None,
                    metrics=metrics,
                )
            )
        
        # Execute tasks in parallel with optional progress bar
        if show_progress:
            results = []
            for coro in tqdm_asyncio.as_completed(
                tasks, 
                total=n, 
                desc="Evaluating dataset"
            ):
                results.append(await coro)
        else:
            results = await asyncio.gather(*tasks)
        
        # Convert results to Polars DataFrame
        df = pl.DataFrame(results)
        return df