python code_generation.py \
  --model_path meta-llama/Llama-3.2-3B-Instruct \
  --input_csv benchmark_v1.csv \
  --framework vllm \
  --num_gpus 1

# python code_generation.py \
#   --model_name meta-llama/Llama-3.2-3B-Instruct \
#   --input_csv benchmark_v1.csv \
#   --framework huggingface \

# python code_generation.py \
#   --model_name gpt-4o-mini \
#   --input_csv benchmark_v1.csv \
#   --framework openai \

