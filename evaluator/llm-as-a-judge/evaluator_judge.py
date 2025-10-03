from Judge import GPTJudge, SeleneJudge
from judge_metrics import compute_context_recall, compute_context_precision
import torch
import argparse
import os
import json

#--------------------Modificar con lógica de cada uno--------------------------
def extract_eval_fields(example):
    """Extract necessary fields from the evaluation example."""
    user_input = example['user_input']
    reference_response = example['reference_answer']
    retrieved_contexts = [context_json['context'] for context_json in example['retrieved_contexts']]
    return user_input, reference_response, retrieved_contexts
#-------------------------------------------------------------------------------

def evaluate_file(results_path, judge_llm, metric="recall"):
    """Evaluate a single results file for the specified metric."""
    with open(results_path) as f:
        eval_dataset = json.load(f)
    metric_scores = []
    for i, example in enumerate(eval_dataset):
        user_input, reference_response, retrieved_contexts = extract_eval_fields(example)
        print(f"--------------Evaluating question {i}: {user_input}-----------------\n")
        if metric == "recall":
            score = compute_context_recall(judge_llm, retrieved_contexts, reference_response)
            print(f"Context Recall: {score:.2f}\n")
        else:
            score = compute_context_precision(judge_llm, retrieved_contexts, user_input, reference_response)
            print(f"Context Precision: {score:.2f}\n")
        metric_scores.append(score)
    avg_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.0
    print(f"Average Context {metric.capitalize()} for {os.path.basename(results_path)}: {avg_score:.3f}")
    return avg_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with LLM-as-Judge Retrieval Results")
    parser.add_argument('--judge_model', type=str, choices=['gpt', 'selene'], default='selene', help='LLM model to use as judge: gpt or selene')
    parser.add_argument('--file', type=str, default=None, help='Path to a single results file')
    parser.add_argument('--folder', type=str, default=None, help='Path to a folder with multiple results files')
    parser.add_argument('--output', type=str, default="context_metric_results.jsonl", help='Output file for folder mode')
    parser.add_argument('--metric', type=str, choices=['recall', 'precision'], default='recall', help='Metric to evaluate: recall or precision')
    parser.add_argument('--cache_dir', type=str, default=None, help='Cache directory for LLM models')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.judge_model == 'gpt':
        judge_llm = GPTJudge(cache_dir=args.cache_dir, device=device)
    elif args.judge_model == 'selene':
        judge_llm = SeleneJudge(cache_dir=args.cache_dir, device=device)
    else:
        raise ValueError("Unsupported judge model. Choose 'gpt' or 'selene'.")

    output_path = args.output
    if args.folder:
        for filename in sorted(os.listdir(args.folder)):
            if filename.endswith('.json'):
                results_path = os.path.join(args.folder, filename)
                print(f"\nEvaluating file: {results_path}")
                avg_score = evaluate_file(results_path, judge_llm, metric=args.metric)
                # Save result after each file
                with open(output_path, "a") as out_f:
                    out_f.write(json.dumps({
                        "file": filename,
                        f"average_context_{args.metric}": avg_score
                    }) + "\n")
    elif args.file:
        avg_score = evaluate_file(args.file, judge_llm, metric=args.metric)
        with open(output_path, "a") as out_f:
                    out_f.write(json.dumps({
                        "file": args.file,
                        f"average_context_{args.metric}": avg_score
                    }) + "\n")
        print(f"Average Context {args.metric.capitalize()}: {avg_score:.3f}")
    else:
        print("Please provide either --file <file> or --folder <folder> argument.")