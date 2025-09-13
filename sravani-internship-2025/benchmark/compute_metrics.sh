python compute_metrics.py \
  --input_path ./LLMjudge_outputs/code_correctness/Llama-3.1-8B-Instruct_results_llmjudge_with_correctness.jsonl \
  --column_name "Constraint_adherence" \
  --output_dir "./metrics_outputs" \
  --metrics_summary_file "./metrics_outputs/metrics_summary.jsonl"
