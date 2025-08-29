python compute_metrics_v2.py \
  --input_path ./LLMjudge_outputs/code_correctness/granite-3.1-2b-instruct_results_correctness.jsonl \
  --column_name "Constraint_adherence" \
  --output_dir "./metrics_outputs/test" \
  --metrics_summary_file "./metrics_outputs/metrics_summary_v2_test.jsonl"
