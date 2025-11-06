import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Reranker():
    def __init__(self, reranker_model):
        """
        Initialize the Reranker with the specified model.
        :param reranker_model: The name or path of the pre-trained reranker model.
        """

        # Load the SentenceTransformer model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model, torch_dtype="auto", trust_remote_code=True)
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model, torch_dtype="auto",trust_remote_code=True)

        self.reranker_model.to(self.device)
        self.reranker_model.eval()

    def rerank_results(self, query, documents, data):
        """
        Rerank the retrieved documents using the loaded reranker model.
        :param query: The input query string.
        :param documents: A list of retrieved document strings.
        :param data: A list of metadata associated with each document.
        :return: A list of tuples containing (document, metadata, score) sorted by score in descending order.
        """
        #reranker_model.eval()  # Ensure model is in evaluation mode

        # Prepare query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Tokenize inputs
        with torch.no_grad():
            inputs = self.reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            
            # Move inputs to the same device as the model (GPU if available)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Compute scores
            scores = self.reranker_model(**inputs, return_dict=True).logits.view(-1).cpu().float().numpy()

        # Combine documents with metadata and scores
        scored_docs = list(zip(documents, data, scores))

        # Sort by score in descending order
        ranked_docs = sorted(scored_docs, key=lambda x: x[2], reverse=True)

        return ranked_docs

