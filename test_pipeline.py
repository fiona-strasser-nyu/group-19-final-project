"""

This script evaluates the LibrAIrian safety pipeline using NVIDIA Aegis AI Content Safety Dataset.

The evaluation does:
1. Input safety filtering so safe vs. unsafe prompt classification
2. Output safety filtering for toxicity, vocabulary, and readability
3. Retrieval quality using embedding similarity
4. Simulated generation latency statistics

This prepares labeled evaluation data, runs each evaluation component, and reports performance metrics for analysis
"""

from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from evaluate_filter import FilterEvaluator

# Load Aegis dataset
# Provide labeled prompts indicating whether content is safe or unsafe
# Used to evaluate input safety filter 
ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
df = ds.to_pandas()
df_subset = df.head(1000)

# Load tokenizer used by BERT toxicity model
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")  # match model used in filter
MAX_TOKENS = 512

# Prepare evaluation data for input filter
test_data = []
for _, row in df_subset.iterrows():
    prompt = row["prompt"] # truncated text input
    # skip invalid or empty prompts
    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        continue
    # truncate by tokens
    tokens = tokenizer(prompt, truncation=True, max_length=MAX_TOKENS, return_tensors=None)
    truncated_prompt = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

    # label of 0 = safe and 1 = unsafe
    # convert dataset label to binary format
    label_raw = row.get("prompt_label", "safe")
    label = 1 if label_raw.lower() == "unsafe" else 0

    test_data.append({"prompt": truncated_prompt, "label": label})

text_file_path = "data/cleaned_merged_fairy_tales_without_eos.txt"

# Initialize Evaluator
evaluator = FilterEvaluator()

# Evaluate Input Filter
# print("Evaluating input filter...")
# input_results = evaluator.evaluate_input_filter(test_data)
# print(input_results)

# print("Evaluating output filter...")
output_results = evaluator.evaluate_output_filter(
    text_file_path, 
    num_samples=100,  # how many prompts to sample
    fk_range=(3,5)    # target FK grade range
)
print(output_results)

# Retrieval Evaluation
print("\nEvaluating retrieval metrics...")
# use prompts that were sampled for output evaluation
output_prompts = evaluator.test_prompts  

# build embedding store for retrieval evaluation
evaluator.build_embedding_store(output_prompts)

# Ground Truth Mapping
# each query is expected to retrieve itself as top relevant chunk
ground_truth = {p: p for p in output_prompts[:10]}  # first 10 prompts
retrieval_results = evaluator.evaluate_retrieval(ground_truth, k=3)
print(retrieval_results)

# Latency Evaluation
print("\nMeasuring latency...")
# measure simulated generation latency on first 50 prompts
latency_results = evaluator.measure_latency(output_prompts[:50])
print(latency_results)
