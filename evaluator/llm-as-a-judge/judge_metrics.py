import re
from prompts import CONTEXT_RECALL_PROMPT, CONTEXT_PRECISION_PROMPT

def split_sentences(text): #Naive approach to split sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if len(s) > 3]

def build_context_recall_prompt(sentence, context):
    return CONTEXT_RECALL_PROMPT.format(
        sentence=sentence,
        context=context)

def build_context_precision_prompt(context, question, ground_truth):
    return CONTEXT_PRECISION_PROMPT.format(
        context=context,
        question=question,
        ground_truth=ground_truth
    )

def compute_context_recall(judge, contexts, ground_truth):
    gt_sentences = split_sentences(ground_truth)
    if not gt_sentences:
        return 0.0
    relevant_count = 0
    for sent in gt_sentences:
        for ctx in contexts:
            prompt = build_context_recall_prompt(sent, ctx)
            #print(prompt)
            result = judge.evaluate(prompt)
            if "yes" in result.lower():
                relevant_count += 1
                break  # One supporting context is enough
    return relevant_count / len(gt_sentences)


def compute_context_precision(judge, contexts, question, ground_truth):
    """
    Compute Context Precision as the mean of precision@k for each chunk in contexts.
    Precision@k = (number of relevant chunks in top k) / k

    - For each context chunk, ask the judge if it is relevant (true positive).
    - For each k (from 1 to N), compute Precision@k.
    - Return the mean of all Precision@k values.

    Args:
        judge: The LLM judge object.
        contexts: List of retrieved context chunks.
        question: The original question.
        ground_truth: The reference answer.

    Returns:
        float: Context Precision score.
    """
    if not contexts:
        return 0.0
    relevance = []
    for ctx in contexts:
        prompt = build_context_precision_prompt(ctx, question, ground_truth)
        result = judge.evaluate(prompt)
        # Consider "yes" as relevant
        is_relevant = "yes" in result.lower()
        relevance.append(is_relevant)
    precisions = []
    relevant_so_far = 0
    for k, rel in enumerate(relevance, start=1):
        if rel:
            relevant_so_far += 1
        # Precision@k: relevant_so_far / k
        precisions.append(relevant_so_far / k)
    # Context Precision: mean of all Precision@k
    return sum(precisions) / len(precisions) if precisions else 0.0
