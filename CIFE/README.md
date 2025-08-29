# Setup
```
pip install -r requirements.txt
```
# Response Generation
## Option 1: Using VLLM
To use vllm for response generation run the command `pip install vllm`

```python
python code_generation.py \
  --model_path <model_path>\
  --framework vllm 
  ```

## Option 2: Using HuggingFace
To use huggingface for response generation run the command `pip install huggingface`
```python
python code_generation.py \
  --model_path <model_path> \
  --framework huggingface 
  ```

## Option 3: Using OpenAI Client

```python
python code_generation.py \
  --model_path <model_path> \
  --framework openai 
  ```
## Option 4 : Others
For generating responses using other than above mentioned 3 methods
- use the column `instruction` as input prompt from the file `benchmark_data.jsonl` 
- The generated responses should be saved with the column name as `response` keeping all the fields in the input file as it is 

# Response Evaluation using LLM Judge 

```python
python llm_judge/llm_judge.py \
    --input_path <input_path> \
    --output_path <output_path> \
    --api_key <api_key> 
```

# Metrics Computation

```python
python compute_metrics.py \
    --input_path <input_path> \
    --metrics_summary_filepath <metrics_summary_filepath> 
```

