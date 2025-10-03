CONTEXT_RECALL_PROMPT = """You are tasked with evaluating whether a specific sentence from a Galician answer is supported by a retrieved context, based on a binary scoring rubric. Provide comprehensive feedback strictly adhering to the rubric, followed by a binary Yes/No judgment. Avoid generating any additional opening, closing, or explanations.

⚠️ Note: The sentence and context are written in Galician. Do not translate or modify the original language. Evaluate as-is.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the factual content of the sentence is present in the retrieved context. The sentence should be considered supported if the same factual content exists in the context, without requiring translation or interpretation.
(2) If the sentence cannot be confirmed based on the context, respond "No".

Your reply should strictly follow this format:
**Result:** <Yes or No>

Here is the data:

Sentence:
{sentence}

Retrieved Context:
{context}

Score Rubrics:
Yes: The factual content of the sentence is clearly present in the retrieved context.
No: The factual content of the sentence is not present or cannot be confirmed in the retrieved context.
"""

CONTEXT_PRECISION_PROMPT = """You are tasked with evaluating whether a specific retrieved context is relevant to answering a Galician question, based on a binary scoring rubric. Provide comprehensive feedback strictly adhering to the rubric, followed by a binary Yes/No judgment. Avoid generating any additional opening, closing, or explanations.

⚠️ Note: The context, question, and ground truth answer are written in Galician. Do not translate or modify the original language. Evaluate as-is.

Here are some rules of the evaluation:
(1) You should prioritize evaluating whether the context clearly helps answer the question. Only respond "Yes" if the sentence is clearly relevant to producing or supporting the ground truth answer.
(2) If the context does not help answer the question or is irrelevant, respond "No".

Your reply should strictly follow this format:

**Result:** <Yes or No>

Here is the data:

Context:
{context}

Question:
{question}

Ground Truth Answer:
{ground_truth}

Score Rubrics:
Yes: The context  clearly helps answer the question and supports the ground truth answer.
No: The context  does not help answer the question or is irrelevant to the ground truth answer.
"""

__all__ = ['CONTEXT_RECALL_PROMPT', 'CONTEXT_PRECISION_PROMPT']