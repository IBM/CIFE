
python run_vllm_model.py --model_name meta-llama/Llama-3.1-8B-Instruct \
  --batch_size 8 \
  --max_new_tokens 1024 \
  --temperature 0 \
  --output_dir outputs \
  --input_csv benchmark_v1.csv \
