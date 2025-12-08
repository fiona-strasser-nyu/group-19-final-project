from rag import KidsRAG
from agent import LibrAIrianAgent, ChildMessagesState
from output_filter import OutputFilter
from input_filter import InputFilter
from llm import LLM

rag = KidsRAG(
    data_path = "data/cleaned_merged_fairy_tales_without_eos.txt",
    passage_size = 120,
    model_name = "all-MiniLM-L6-v2",
    output_dir = "output_data"
)

rag.prepare_data()

# debug

print(rag.passages_df.head())
print(rag.passages_df.columns)

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