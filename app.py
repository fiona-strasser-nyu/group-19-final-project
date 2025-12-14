import streamlit as st
from input_filter import InputFilter
from output_filter import OutputFilter
from rag import KidsRAG
from llm import LLM
from agent import LibrAIrianAgent, ChildMessagesState
import os
from pathlib import Path

"""
app.py

This file defines the Streamlit application for Project LibrAIrian, which is the interactive interface.
This child reading assistant allows users to ask safe, guided questions about stories using a
retrieval-augmented generation (RAG) pipeline.

The application
1. Loads and caches RAG, LLM, safety filters
2. Handles OpenAI API key input
3. Manages user interaction and session state
4. Sends user queries to the LibrAIrian agent
5. Displays a chat style interface for responses
"""

st.title("LibAIrian Resource Center")

st.markdown("""
Welcome to the Project LibrAIrian library! We want to help you better understand 
and think about the stories you read, while having fun and staying safe. 

Your "librarian" is a virtual assistant with lots of knowledge about stories through
all the time they've spent looking and training on fairy tales.

You can ask a question about one of those stories, a different story, or a general
question. Just say what you're wondering about! To see what stories we have specific
knowledge about, take a look at the sidebar.

Have fun! Keep reading!
""")

# OpenAI API Key Handling
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
  api_key = st.text_input("Enter your OpenAI API key", type="password")
  if not api_key:
    st.stop()

@st.cache_resource(show_spinner=True)
def load_rag():
  """
    Initialize and return a KidsRAG object

    This function loads and prepares the fairy tale dataset used for RAG.

    Returns:
        KidsRAG: An initialized RAG object for passage retrieval.
  """
  rag = KidsRAG(
        data_path = "data/cleaned_merged_fairy_tales_without_eos.txt",
        passage_size = 120,
        model_name = "all-MiniLM-L6-v2",
        output_dir = "output_data"
    )
  return rag

rag = load_rag()

embeddings_file = Path("output_data/embeddings.npy")
passages_file = Path("output_data/passages.pkl")

rag_loaded = False
if embeddings_file.exists() and passages_file.exists():
  rag.load_saved_passages()
  rag.load_saved_embeddings()
  rag_loaded = True

if not rag_loaded:
  st.write("Resource data is not available. Click the button to 'stock the library shelves'!")
  if st.button("Generate resources now"):
    with st.spinner("One moment..."):
      rag.prepare_data()
      rag_loaded = True

if rag_loaded == False:
  st.stop()

# load safety filter, llm, and agent
with st.sidebar:
  st.header("Reference Texts in this Library")

  if hasattr(rag, "passages_df") and rag.passages_df is not None:
    titles = sorted(rag.passages_df['title'].unique())
    for title in titles:
      st.markdown(title)

@st.cache_resource(show_spinner=True)
def load_input_filter():
  """
    Loads and returns the input safety filter.

    This filter checks whether user queries are appropriate and safe for children.

    Returns:
        InputFilter: Initialized input safety filter
  """
  return InputFilter(threshold = 0.2)

@st.cache_resource(show_spinner=True)
def load_output_filter():
  """
    Load and return the output safety filter.

    This filter evaluates LLM-generated responses for toxicity, inappropriate topics, and reading level suitability.

    Returns:
        OutputFilter: Initialized output safety filter
  """
  output_filter = OutputFilter(
      toxic_threshold = 0.2,
      topic_threshold = 0.6,
      dale_chall_file = "dale_chall_words.txt"
  )
  return output_filter

@st.cache_resource(show_spinner=True)
def load_llm(api_key):
  """
    Initializes and returns the LLM wrapper.

    Args: api_key (str), OpenAI API key.

    Returns:
        LLM: Initialized language model interface
  """
  llm = LLM(
    model = "gpt-4o-mini",
    max_tokens = 300,
    temperature = 0.7,
    api_key = api_key
  )

  return llm

@st.cache_resource(show_spinner=True)
def load_agent(_rag, _llm, _input_filter, _output_filter):
  """
    Create and return a cached agent

    This agent coordinates the RAG pipeline, LLM calls, safety filtering, and conversation flow using LangGraph.

    Args:
        _rag (KidsRAG): RAG system for passage retrieval
        _llm (LLM): Language model wrapper
        _input_filter (InputFilter): User input safety filter
        _output_filter (OutputFilter): Output safety filter

    Returns:
        LibrAIrianAgent: Initialized conversational agent.
  """
  agent = LibrAIrianAgent(
      rag = _rag,
      llm = _llm,
      input_filter = _input_filter,
      output_filter = _output_filter,
      max_turns = 10
  )

  return agent

input_filter = load_input_filter()
output_filter = load_output_filter()
llm = load_llm(api_key)
agent = load_agent(rag, llm, input_filter, output_filter)

# initialize session states
if "messages" not in st.session_state:
  st.session_state.messages = []
if "turn_count" not in st.session_state:
  st.session_state.turn_count = 0
if "story_title" not in st.session_state:
  st.session_state.story_title = ""
if "user_query" not in st.session_state:
  st.session_state.user_query = ""

# user inputs
story_title = st.text_input(
    "What story are you asking about? (Type 'general' if it is a general question):",
    st.session_state.story_title
)

st.session_state.story_title = story_title

def ask_question():
  """
    Send the user's query through the LibrAIrian agent and update the Streamlit session state with the response.

    This function constructs a ChildMessagesState object, 
    invokes the LangGraph agent, and then stores the resulting
    messages and turn count so the conversation can continue
  """
  user_query = st.session_state.input_box

  state = ChildMessagesState(
        story_title = story_title,
        user_query = user_query,
        query_type = 2,
        messages=st.session_state.messages.copy(),
        retrieve_passages=None,
        response="",
        final_output="",
        turn_count=st.session_state.turn_count
    )

  answer = agent.graph.invoke(state)

  st.session_state.messages = answer.get("messages", st.session_state.messages)
  st.session_state.turn_count = answer.get("turn_count", st.session_state.turn_count)

  st.session_state.user_query = ""

user_query = st.text_input("How can LibrAIrian help you? (Type 'quit' to exit):", key="input_box")

if st.button("Ask"):
  ask_question()
  st.rerun()

# display chat
# AI disclaimer: used it to figure out how to format with this appearance
for msg in st.session_state.messages:
  if msg["role"] == 'user':
    st.markdown(f"""
      <div style="
          background-color:#DCF8C6;
          padding:10px;
          border-radius:10px;
          text-align:right;
          margin: 5px;
          display:inline-block;
          max-width:70%;
          word-wrap: break-word;">
          {msg['content'].replace('\n', '<br>')}
      </div>
      """, unsafe_allow_html=True)
  else:
    st.markdown(f"""
        <div style="
            background-color:#FFFFFF;
            padding:10px;
            border-radius:10px;
            text-align:left;
            margin: 5px;
            display:inline-block;
            max-width:70%;
            word-wrap: break-word;
            border:1px solid #ECECEC;">
            {msg['content'].replace('\n', '<br>')}
        </div>
        """,
        unsafe_allow_html=True)

# st.text_input("answer['final_output'])
