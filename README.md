# group-19-final-project

**Topic:** Child-Friendly AI to Serve as Early Reading Assistant
        
**Research question:** How can an AI assistant help with critical engagement and comprehension in young readers while ensuring emotional safety, age-appropriate interaction, and limited screen dependency?
        
**Significance:** Generic AI may generate text beyond children's comprehension level, fail to restrict mature content, and discourage independent attempts at critical thinking through providing plain summary. We aim to build a chatbot that can help children learn without risking intellectual or emotional health. 

**Structure**
In folders milestone_1, milestone_2, and milestone_3, we have the code used to update or experiment throughout the process of building our agent. This includes demos of the RAG + LLM and the safety filtering layers before integration, including evaluation and analysis, In milestone_3, this includes this same analysis on the full model using the graph structure. The code documents in these folders may not run within the structure as is as they depend on files outside of their current folder which are part of the main framework.

In visuals, we have the results of our EDA and the results of different evaluations (confusion matrices, bar graphs).

In data, we have the dataset used for our RAG and the list of titles. 

In the main repository, we have four main files holding the classes that define our model: rag.py, llm.py, input_safety.py, output_safety.py, and agent.py. These are the files needed to createa our model. To run our streamlit UI, use app.py

We also have two files that were used for testing: evalute_filter.py, which defines the process of going through the safety filters, and test_pipeline, which actually uses this to evalutate our safety filtering quality.

**Tutorial:** To run the streamlit app from colab:

```
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/[your path]

!pip install -r requirements.txt

from pyngrok import ngrok

NGROK_AUTH_TOKEN = input("Enter your ngrok auth token:")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

!pkill streamlit
!pkill ngrok

ngrok.kill()

!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &

public_url = ngrok.connect((8501))
print("Ngrok URL:", public_url)
```
You will need an ngrok auth token to create the link. Then open the generated link.

Otherwise, to run the app in the terminal, make sure you have installed streamlit. Then run:

```
cd /[your path to the project]

pip install -r requirements.txt

streamlit run my_app.py
```
Once the streamlit is open, you will be prompted to enter your OpenAI API key.

If this is your first time running the system and there are no saved embeddings, click "Generate Resources" for the RAG to load the passages and create embeddings.

In the first field, enter the story title you have a question about (either in the database or not--we will check!) or "general" for a general query. Ask you query 
in the second text box. Click "Ask" to generate a response. Keep asking questions until you reach the limit or you are done, at which point you can enter "quit" to
end the session.
