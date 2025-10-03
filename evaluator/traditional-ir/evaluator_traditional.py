from ir_metrics import compute_mrr, compute_precision, compute_recall
import argparse
import json
import os

#--------------------Modificar con lógica de cada uno--------------------------
def extract_eval_fields(example, level='paragraph'):
    """Extract necessary fields from the evaluation example."""
    if level == 'paragraph':
        reference_sources = [f"{example['reference_source_id']}-{ref_paragraph}" 
                                 for ref_paragraph in example['reference_context_paragraphs']]
        retrieved_sources = [f"{ctx['context_metadata']['source_id']}-{ctx['context_metadata']['paragraph_position']}" 
                                 for ctx in example['retrieved_contexts']]
    else:  # document level
        reference_sources = [example['reference_source_id']]
        retrieved_sources = [f"{ctx['context_metadata']['source_id']}" 
                                 for ctx in example['retrieved_contexts']]                             
    return reference_sources, retrieved_sources
#-------------------------------------------------------------------------------

def evaluate_retrieval(eval_dataset, level='paragraph'):
    results = {
        'precision': [],
        'recall': [],
        'mrr': []
    }
    
    deduplicate = True if level == 'document' else False # If document level, deduplicate by source_id. In paragraph level, do not deduplicate.

    for eval_item in eval_dataset:
        reference_sources, retrieved_sources = extract_eval_fields(eval_item, level=level)
        precision = compute_precision(reference_sources, retrieved_sources, deduplicate=deduplicate)
        recall = compute_recall(reference_sources, retrieved_sources, deduplicate=deduplicate)
        mrr = compute_mrr(reference_sources, retrieved_sources)

        results['precision'].append(precision)
        results['recall'].append(recall)
        results['mrr'].append(mrr)
    
    # Calculate averages
    avg_results = {
        'avg_precision': sum(results['precision']) / len(results['precision']) if results['precision'] else 0,
        'avg_recall': sum(results['recall']) / len(results['recall']) if results['recall'] else 0,
        'avg_mrr': sum(results['mrr']) / len(results['mrr']) if results['mrr'] else 0
    }
    return avg_results

def evaluate_file(results_path):
    with open(results_path) as f:
        eval_dataset = json.load(f)
    results_paragraph = evaluate_retrieval(eval_dataset, level='paragraph')
    results_document = evaluate_retrieval(eval_dataset, level='document')
    return {
        "file": os.path.basename(results_path),
        "avg_precision_paragraph": results_paragraph["avg_precision"],
        "avg_recall_paragraph": results_paragraph["avg_recall"],
        "avg_mrr_paragraph": results_paragraph["avg_mrr"],
        "avg_precision_document": results_document["avg_precision"],
        "avg_recall_document": results_document["avg_recall"],
        "avg_mrr_document": results_document["avg_mrr"]
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with traditional metrics Retrieval Results")
    parser.add_argument('--file', type=str, default=None, help='Path to a single results file')
    parser.add_argument('--folder', type=str, default=None, help='Path to a folder with multiple results files')
    parser.add_argument('--output', type=str, default="traditional_metric_results.jsonl", help='Output file for folder mode')
    parser.add_argument('--logging', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    if args.folder:
        output_path = args.output
        for filename in sorted(os.listdir(args.folder)):
            if filename.endswith('.json'):
                results_path = os.path.join(args.folder, filename)
                print(f"Evaluating file: {results_path}")
                result = evaluate_file(results_path)
                with open(output_path, "a") as out_f:
                    out_f.write(json.dumps(result) + "\n")
    elif args.file:
        result = evaluate_file(args.file, logging=args.logging)
        print(json.dumps(result, indent=2))
    else:
        print("Please provide either --file <file> or --folder <folder> argument.")