import re
import requests
from utils.utils import today
import os
import json
from openai import OpenAI
from datetime import datetime
from reranker import Reranker
from sentence_transformers import SentenceTransformer
import copy

class RAG:
    def __init__(self, config_path, indices_config):
        """
        Initialize the RAG system with configurations from a JSON file.
        :param config_path: Path to the configuration JSON file.
        :param indices_config: Path to the indices configuration JSON file.
        """

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Error: No se ha encontrado el archivo de configuracion {config_path}")

        if not os.path.exists(indices_config):
            raise FileNotFoundError(f"Error: No se ha encontrado el archivo de configuracion {indices_config}")
        
        with open(config_path, "r", encoding="utf-8") as archivo:
            config_json = json.load(archivo) 

        try:
            # load configurations from JSON file, if there is an error, exit
            self.retrieval_method = config_json["RETRIEVAL_METHOD"]  

            # PROMPTS
            prompts_json = config_json["PROMPTS"]
            self.SYSTEM_PROMPT = prompts_json["retrieval_system_prompt"]
            self.SYSTEM_PROMPT_NO_RETRIEVAL = prompts_json["non_retrieval_system_prompt"]

            # NUMBER OF DOCUMENTS TO RETRIEVE AND RERANK
            self.NUM_DOCS_RETIEVAL = config_json["NUM_DOCS_RETIEVAL"]
            self.NUM_DOCS_RERANKER = config_json["NUM_DOCS_RERANKER"]
            self.chunk_size = config_json["CHUNK_SIZE"]
            self.reranker_model_path = config_json["RERANKER_MODEL"]
            self.embedding_model = config_json["EMBEDDING_MODEL"]
            self.MODEL = config_json["MODEL"]  # default model
            self.MODELS = config_json["MODELS"]  # List of available models

            # elasticsearch (default index)
            elastic_search_config = config_json["elastic_search"]
            self.es_host = elastic_search_config["host"]
            self.es_port = elastic_search_config["port"]

            # Load index configurations from JSON file
            with open(indices_config, "r", encoding="utf-8") as archivo:
                self.index_configs = json.load(archivo)
                self.available_indices = list(self.index_configs.keys())

        except json.JSONDecodeError:
            print(f"Archivo de configuracion no valido")
            exit(1)

        self.MODEL_SYSTEM_PROMPT = self.SYSTEM_PROMPT  # Default system prompt
        self.reranker_model = Reranker(self.reranker_model_path)

        self.available_model_names = list(self.MODELS.keys())

        # embedding model to encode the query for elastic search (if needed)
        self.encodermodel = SentenceTransformer(self.embedding_model)

        # Initialize the model configuration
        self.change_model(self.MODEL)  # Set the initial model


    def change_model(self, model_name):
        """
        Change the model used for RAG.
        :param model_name: Name of the model to switch to.
        """
        
        if model_name not in self.MODELS:
            raise ValueError(f"Model {model_name} not found in available models.")
        
        self.MODEL = model_name
        self.openai_base_url = self.MODELS[self.MODEL]["openai_base_url"]
        self.openai_api_key = self.MODELS[self.MODEL]["openai_api_key"]
        self.MODEL_NAME = self.MODELS[self.MODEL]["model_name"]

        # Reinitialize the OpenAI client with the new model
        self.client = OpenAI(base_url=self.openai_base_url, api_key=self.openai_api_key)


    def generate_response(self, chat_history, model, domain, retrieval_method, top_k):
        """
        Generate a response for the user query and selected domain and method
        :param chat_history: List of messages in the chat history.
        :param model: Model to use for generation.
        :param domain: Domain to use for retrieval.
        :param retrieval_method: Retrieval method to use ("BM25", "Embeddings", "No").
        :param top_k: Number of top documents to use for context.
        :yield: Updated chat history and context for streaming.
        """
        current_user_input = chat_history[-1]["content"] # Get last user message
        message_list = copy.deepcopy(chat_history)  # Deep copy of chat history to modify
        context_with_date = "" # Initialize context with date
        
        # check if the retrieval method has changed
        if retrieval_method != self.retrieval_method:
            self.retrieval_method = retrieval_method
            if self.retrieval_method in ["BM25", "Embeddings"]:
                self.MODEL_SYSTEM_PROMPT = self.SYSTEM_PROMPT
            elif self.retrieval_method == "No":
                self.MODEL_SYSTEM_PROMPT = self.SYSTEM_PROMPT_NO_RETRIEVAL

        if retrieval_method in ["BM25", "Embeddings"]:

            # SOLO VISUALIZACION
            chat_history.append({"role": "assistant", "content": "Buscando información..."})
            yield (chat_history, context_with_date)

            n_docs_to_retrieve = self.NUM_DOCS_RETIEVAL
            n_docs_to_rerank = top_k

            # RETRIEVAL
            query_results = self.search_documents(current_user_input, n_docs_to_retrieve, embeddings=retrieval_method=="Embeddings", index=domain)
            
            documents = query_results["documents"][0]
            metadata = query_results["metadatas"][0]

            # RERANKER
            if len(documents) != 0:
                ranked_docs = self.reranker_model.rerank_results(current_user_input, documents, metadata)
            else:
                print("No documents found.")
                ranked_docs = []

            # Prepare context for the LLM (top 5 documents)
            top_docs = ranked_docs[:n_docs_to_rerank]

            # format context
            model_context_with_date = "\n".join(
                [f"- [{i+1}] : {doc}" for i, (doc, meta, _) in enumerate(top_docs)]
            )
            context_with_date = []
            for i, (doc, meta, _) in enumerate(top_docs):
                context_with_date.append(
                    {
                        "title": meta['title'],
                        "text": meta['text'],
                        "date": meta['date'][:10],
                        "url": meta['url'],
                        "num": i+1
                    }
                )
            context_with_date = json.dumps(context_with_date, ensure_ascii=False)
            user_message = f"Context:\n- {model_context_with_date}\n\nQuestion: {current_user_input}\n\nAnswer:"
            message_list[-1]["content"] = user_message  # Update the last user message with context

        elif retrieval_method == "No":
            user_message = current_user_input
            # SOLO VISUALIZACION
            chat_history.append({"role": "assistant", "content": ""})
            yield (chat_history, context_with_date)
        
        # add system message
        sysmessage = {"role": "system", "content": self.MODEL_SYSTEM_PROMPT}
        message_list = [sysmessage] + message_list
        print("Message list to model:", message_list)
        # GENERATION
        response_stream = self.chat_with_model(message_list) # Stream the response

        chat_history[-1]["content"] = ""
        yield (chat_history, context_with_date) # Yield for streaming, use raw user message for display

        # Stream the response and update the chat history
        for chunk in response_stream:
            if chunk.choices[0].delta.content:
                chat_history[-1]["content"] += chunk.choices[0].delta.content
            yield (chat_history, context_with_date)


    def chat_with_model(self, messages):
        """
        Chat with the LLM using the API and stream the response.
        :param messages: List of messages to send to the model.
        :return: Response from the model.
        """

        response = self.client.chat.completions.create(
            model=self.MODEL_NAME,
            messages=messages,
            max_tokens=512,
            temperature=0.8,
            top_p=0.9,
            stream=True
        )
        return response
    

    def search_documents(self, query, n_docs=20, embeddings=False, index=None):
        """
        Search for documents in Elasticsearch using BM25 or Embeddings.
        :param query: The search query.
        :param n_docs: Number of documents to retrieve.
        :param embeddings: (Bool) Whether to use embeddings for the search.
        :param index: Elasticsearch index to use (None returns empty search).
        :return: Search results from Elasticsearch.
        """

        index_cfg = self.index_configs[index]
        search_index = index_cfg["bm25_index"] if not embeddings else index_cfg["embedding_index"]
   
        # Search query without embeddings (BM25)
        if not embeddings:
            data = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": index_cfg["search_fields"],
                        "type": "cross_fields",
                        #"boost": 0.7
                    }
                },
                "highlight": {
                    "fields": {
                        index_cfg["highlight_field"]: {
                            "fragment_size": self.chunk_size,
                            "number_of_fragments": 1
                        }
                    }
                }
            }

        # search query with embeddings (KNN)
        else:
            data = {
                "size": n_docs,
                "knn": {
                    "field": index_cfg["embedding_field"],
                    "query_vector": self.encodermodel.encode(query).tolist(),
                    "k": 100,
                    "num_candidates": 1000
                }
            }

        url_search = f"{self.es_host}:{self.es_port}/{search_index}/_search?size={n_docs}"

        try:
            response = requests.post(url_search, json=data)
            response.raise_for_status()
            
            hits = response.json()["hits"]["hits"]

            return self.format_response(hits, index, embeddings)
            
        except requests.exceptions.RequestException as e:
            print(f"Error en la búsqueda: {e}")
            return {"documents": [], "metadatas": [], "scores": []}
        

    def format_response(self, hits, index, embeddings):
        """
        Format the Elasticsearch response to adapt it to the RAG system, reranker and output. Format will be done according to the configuration file.
        :param hits: Hits from Elasticsearch response.
        :param index: index to use for formatting.
        :return: Formatted documents, metadatas, and scores.
        """

        index_cfg = self.index_configs[index]
        documents, metadatas, scores = [], [], []
        
        for hit in hits:
            source = hit["_source"]
            highlight_field = index_cfg["highlight_field"]

            if embeddings:
                highlighted_fragments = [source.get(index_cfg["embedding_text_field"], "")]
            else:
                highlighted_fragments = hit.get("highlight", {}).get(highlight_field, [""])
            
            for contenido_texto in highlighted_fragments:
                format_template = index_cfg["formatted_text"]
                context = {field: source.get(field, "") for field in index_cfg["formatted_text_fields"]}
                context.update({"text": contenido_texto})
                formatted_text = format_template.format_map(context)
                documents.append(formatted_text)

                metadatas.append({
                    **{field: source.get(src_field, "") for field, src_field in index_cfg["metadata_fields"].items()},
                    "text": contenido_texto,
                })

                scores.append(hit.get("_score", 0))

        return {"documents": [documents], "metadatas": [metadatas], "scores": [scores]}

        