
import streamlit as st
from input_filter import InputFilter
from output_filter import OutputFilter
from rag import KidsRAG
from llm import LLM
from agent import LibrAIrianAgent, ChildMessagesState
import os
from pathlib import Path

st.title("LibAIrian Resource Center")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
  api_key = st.text_input("Enter your OpenAI API key", type="password")
  if not api_key:
    st.stop()

@st.cache_resource(show_spinner=True)
def load_rag():
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

@st.cache_resource(show_spinner=True)
def load_input_filter():
  return InputFilter(threshold = 0.2)

@st.cache_resource(show_spinner=True)
def load_output_filter():
  output_filter = OutputFilter(
      toxic_threshold = 0.2,
      topic_threshold = 0.6,
      dale_chall_file = "dale_chall_words.txt"
  )
  return output_filter

@st.cache_resource(show_spinner=True)
def load_llm(api_key):
  llm = LLM(
    model = "gpt-4o-mini",
    max_tokens = 300,
    temperature = 0.7,
    api_key = api_key
  )

  return llm

@st.cache_resource(show_spinner=True)
def load_agent(_rag, _llm, _input_filter, _output_filter):
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

if "messages" not in st.session_state:
  st.session_state.messages = []
if "turn_count" not in st.session_state:
  st.session_state.turn_count = 0
if "story_title" not in st.session_state:
  st.session_state.story_title = ""
if "user_query" not in st.session_state:
  st.session_state.user_query = ""

story_title = st.text_input(
    "What story are you asking about? (Type 'general' if it is a general question):",
    st.session_state.story_title
)

st.session_state.story_title = story_title.strip()

def ask_question():
  user_query = st.session_state.input_box

  state = ChildMessagesState(
        story_title = story_title.strip(),
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
