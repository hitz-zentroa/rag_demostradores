from typing import List

def compute_mrr(expected_ids, retrieved_ids):
    """mrr metric.
    Args:
    expected_ids List[str]: The ground truth node_id
    retrieved_ids List[str]: The node_id from retrieved chunks

    Returns:
        float: MRR score as a decimal
    """
    if retrieved_ids is None or expected_ids is None:
        raise ValueError("Retrieved ids and expected ids must be provided")
    for i, id in enumerate(retrieved_ids):
        if id in expected_ids:
            return (1.0 / (i + 1))
    return (0.0)

# ...existing code...

def compute_precision(expected_ids: List[str], retrieved_ids: List[str], k: int = None, deduplicate: bool = False) -> float:
    """Compute precision metric at k (optional).
    
    Args:
        expected_ids (List[str]): The ground truth node_ids
        retrieved_ids (List[str]): The node_ids from retrieved chunks
        k (int, optional): Number of retrieved results to consider. If None, use all.
    
    Returns:
        float: Precision score as a decimal between 0 and 1
    """
    if not retrieved_ids or not expected_ids:
        return 0.0
    if deduplicate:
        retrieved_ids = list(dict.fromkeys(retrieved_ids))  # preserves order, removes duplicates
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
        
    relevant_retrieved = sum(1 for id in retrieved_ids if id in expected_ids)
    return relevant_retrieved / len(retrieved_ids)

def compute_recall(expected_ids: List[str], retrieved_ids: List[str], k: int = None, deduplicate: bool = False) -> float:
    """Compute recall metric at k (optional).
    
    Args:
        expected_ids (List[str]): The ground truth node_ids
        retrieved_ids (List[str]): The node_ids from retrieved chunks
        k (int, optional): Number of retrieved results to consider. If None, use all.
    
    Returns:
        float: Recall score as a decimal between 0 and 1
    """
    if not retrieved_ids or not expected_ids:
        return 0.0
    if deduplicate:
        retrieved_ids = list(dict.fromkeys(retrieved_ids))  # preserves order, removes duplicates
    if k is not None:
        retrieved_ids = retrieved_ids[:k]
        
    relevant_retrieved = sum(1 for id in retrieved_ids if id in expected_ids)
    return relevant_retrieved / len(expected_ids)
