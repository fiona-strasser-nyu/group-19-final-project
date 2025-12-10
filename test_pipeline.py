from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd
from evaluate_filter import FilterEvaluator

# Load Aegis dataset
ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", split="train")
df = ds.to_pandas()
df_subset = df.head(1000)

# print(df.columns)
# print(df.head())

# Load the tokenizer used by your BERT model
tokenizer = AutoTokenizer.from_pretrained("unitary/toxic-bert")  # match the model used in your filter
MAX_TOKENS = 512

# Convert to expected format
test_data = []
for _, row in df_subset.iterrows():
    prompt = row["prompt"]
    if not isinstance(prompt, str) or len(prompt.strip()) == 0:
        continue
    # Truncate by tokens
    tokens = tokenizer(prompt, truncation=True, max_length=MAX_TOKENS, return_tensors=None)
    truncated_prompt = tokenizer.decode(tokens["input_ids"], skip_special_tokens=True)

    label_raw = row.get("prompt_label", "safe")
    label = 1 if label_raw.lower() == "unsafe" else 0

    test_data.append({"prompt": truncated_prompt, "label": label})

text_file_path = "data/cleaned_merged_fairy_tales_without_eos.txt"

# Initialize evaluator
evaluator = FilterEvaluator()

# Evaluate input filter
# print("Evaluating input filter...")
# input_results = evaluator.evaluate_input_filter(test_data)
# print(input_results)

# print("Evaluating output filter...")
output_results = evaluator.evaluate_output_filter(
    text_file_path, 
    num_samples=100,  # how many prompts to sample
    fk_range=(3,5)    # target Flesch-Kincaid grade range
)
print(output_results)

print("\nEvaluating retrieval metrics...")
# Use the prompts that were sampled for output evaluation
output_prompts = evaluator.test_prompts  

evaluator.build_embedding_store(output_prompts)

ground_truth = {p: p for p in output_prompts[:10]}  # first 10 prompts
retrieval_results = evaluator.evaluate_retrieval(ground_truth, k=3)
print(retrieval_results)

print("\nMeasuring latency...")
# Test latency on first 50 prompts to keep it fast
latency_results = evaluator.measure_latency(output_prompts[:50])
print(latency_results)

# from rag import KidsRAG
# from agent import LibrAIrianAgent, ChildMessagesState
# from output_filter import OutputFilter
# from input_filter import InputFilter
# from llm import LLM

# # Initialize RAG
# rag = KidsRAG(
#     data_path="data/cleaned_merged_fairy_tales_without_eos.txt",
#     passage_size=120,
#     model_name="all-MiniLM-L6-v2",
#     output_dir="output_data"
# )

# # Lowercase approved titles for matching
# rag.approved_titles = [t.lower() for t in rag.approved_titles]
# print("Approved titles (normalized):", rag.approved_titles)

# # Read raw text from file
# with open(rag.data_path, "r", encoding="utf-8") as f:
#     raw_text = f.read()

# # Chunk data
# df = rag.chunk_data(raw_text)

# # Show what we got
# print("Passages DataFrame head:")
# print(df.head())
# print("Passages DataFrame columns:")
# print(df.columns)
# print("Number of passages:", len(df))

# llm = LLM()

# input_filter = InputFilter()
# output_filter = OutputFilter()

# agent = LibrAIrianAgent(
#     rag=rag,
#     llm=llm,
#     input_filter=input_filter,
#     output_filter=output_filter,
#     max_turns=10
# )

# tests = [
#     {"story_title": "general", "user_query": "What is the moral of the story?"},
#     {"story_title": "The Happy Prince", "user_query": "Who is the main character?"},
#     {"story_title": "Nonexistent Title", "user_query": "Tell me about this story."}
# ]

# for test in tests:
#     state = ChildMessagesState(
#         story_title=test["story_title"],
#         user_query=test["user_query"],
#         query_type=2,  # will get updated in agent
#         messages=[],
#         retrieve_passages=None,
#         response="",
#         final_output="",
#         turn_count=0
#     )
    
#     output = agent.graph.invoke(state)
#     print(f"Story Title: {test['story_title']}")
#     print(f"User Query: {test['user_query']}")
#     print(f"Agent Output:\n{output['final_output']}\n{'-'*50}\n")