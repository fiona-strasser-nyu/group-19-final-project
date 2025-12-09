from rag import KidsRAG
from agent import LibrAIrianAgent, ChildMessagesState
from output_filter import OutputFilter
from input_filter import InputFilter
from llm import LLM

# Initialize RAG
rag = KidsRAG(
    data_path="data/cleaned_merged_fairy_tales_without_eos.txt",
    passage_size=120,
    model_name="all-MiniLM-L6-v2",
    output_dir="output_data"
)

# Lowercase approved titles for matching
rag.approved_titles = [t.lower() for t in rag.approved_titles]
print("Approved titles (normalized):", rag.approved_titles)

# Read raw text from file
with open(rag.data_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# Chunk data
df = rag.chunk_data(raw_text)

# Show what we got
print("Passages DataFrame head:")
print(df.head())
print("Passages DataFrame columns:")
print(df.columns)
print("Number of passages:", len(df))

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