#!/bin/bash

RESULTS_DIR=""
RESULTS_FILE=""
CACHE_DIR=""

OUT_TRAD="$RESULTS_DIR/traditional_metric_results.jsonl"
OUT_RECALL="$RESULTS_DIR/judge_recall.jsonl"
OUT_PRECISION="$RESULTS_DIR/judge_precision.jsonl"

echo "Procesando directorio: $RESULTS_DIR"

cd traditional-ir/
python3 evaluator_traditional.py --folder "$RESULTS_DIR" --output "$OUT_TRAD"

cd ../llm-as-a-judge
python3 evaluator_judge.py --cache_dir "$CACHE_DIR" --file "$RESULTS_FILE" --output "$OUT_RECALL" --metric recall
python3 evaluator_judge.py --cache_dir "$CACHE_DIR" --file "$RESULTS_FILE" --output "$OUT_PRECISION" --metric precision

cd ..
echo "Evaluación terminada."