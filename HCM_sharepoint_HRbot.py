import streamlit as st
import os
import openai
from azure.search.documents import SearchClient
from azure.search.documents.models import QueryType
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import RawVectorQuery
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

st.title("HCM  Assistant")

from azure.core.credentials import AzureKeyCredential
from openai import AzureOpenAI

load_dotenv()

knowledge_base = st.sidebar.selectbox("Choose Knowledge Base", ["Sales & Marketing", "Operations","ClientOperations"])

if knowledge_base == "Sales & Marketing":
    AZURE_COGNITIVE_SEARCH_SERVICE_NAME = "hcmaisearch"
    AZURE_COGNITIVE_SEARCH_INDEX_NAME = "hcmaisearchsm"
    AZURE_COGNITIVE_SEARCH_API_KEY = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY_SM")
    AZURE_COGNITIVE_SEARCH_ENDPOINT = "https://hcmaisearch.search.windows.net"
    azure_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)

if knowledge_base == "Operations":
    AZURE_COGNITIVE_SEARCH_SERVICE_NAME = "hcm-ai-search"
    AZURE_COGNITIVE_SEARCH_INDEX_NAME = "hcmindex"
    AZURE_COGNITIVE_SEARCH_API_KEY_OP = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY_OPERATIONS")
    AZURE_COGNITIVE_SEARCH_ENDPOINT = "https://hcm-ai-search.search.windows.net"
    azure_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY_OP)

if knowledge_base == "ClientOperations":
    AZURE_COGNITIVE_SEARCH_SERVICE_NAME = "hcmaisearch"
    AZURE_COGNITIVE_SEARCH_INDEX_NAME = "hcmaisearchco"
    AZURE_COGNITIVE_SEARCH_API_KEY = os.getenv("AZURE_COGNITIVE_SEARCH_API_KEY_SM")
    AZURE_COGNITIVE_SEARCH_ENDPOINT = "https://hcmaisearch.search.windows.net"
    azure_credential = AzureKeyCredential(AZURE_COGNITIVE_SEARCH_API_KEY)


# Determine the appropriate vector field name based on the selected knowledge base
if knowledge_base == "Operations":
    VECTOR_FIELD_NAME = "vector"
else:
    VECTOR_FIELD_NAME = "embedding"




OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_ENDPOINT = "https://hcmchaos.openai.azure.com/"
OPENAI_API_VERSION = "2024-12-01-preview"
GPT4O = "gpt-4o"
EMBEDDING_MODEL_DEPLOYEMENT_NAME = "embedding-model"



#logo_url = "https://www.ibc.com/images/ibc-logo.png"
logo_url = "https://hcmar.com/wp-content/uploads/2024/05/HCM-Logo-TM-Tagline-Chaos-Full-Color-FNL-1.png"
logo_html = f'<img src="{logo_url}" alt="Logo" height="130" width="250">'
st.sidebar.markdown(f'<div class="logo-container">{logo_html}</div>', unsafe_allow_html=True)


#############################################################################################
def authenticate(password):
   # Replace with your authentication logic
   # For simplicity, a hardcoded email and password are used
   valid_password = "app123"
    
   return password == valid_password
# Session state initialization
if 'logged_in' not in st.session_state:
   st.session_state.logged_in = False
# Login Page
login = st.sidebar.checkbox("Login")
if login and not st.session_state.logged_in:
   st.sidebar.title("Login")
   #email = st.sidebar.text_input("Email")
   password = st.sidebar.text_input("Password", type="password")
   if st.sidebar.button("Login"):
       if authenticate(password):
           st.session_state.logged_in = True
           st.experimental_rerun()
       else:
           st.sidebar.error("Invalid password")
# Check if the user is logged in before proceeding
if not st.session_state.logged_in:
   st.warning("Please log in to use the Assistant.")
   st.stop()  # Stop further execution if not logged in
    
######################################################### Neom ##########################################################    
use_memory = True
if use_memory:
    #st.session_state.messages = []
    if st.sidebar.button(':red[Clear History]'):
        st.session_state.memory_messages = []
        st.session_state.messages = []

######################################################### Neom ##########################################################    
    
if "memory_messages" not in st.session_state:
    st.session_state.memory_messages = []
    
if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        if message["role"] == "assistant":
            avatar = "🤖"
        else:
            avatar = "🧑‍💻"
        with st.chat_message(message["role"],avatar = avatar ):
            st.markdown(message["content"])
######################################################### Neom ##########################################################

######################################################### Neom ##########################################################
import os
from openai import AzureOpenAI

client = AzureOpenAI(
  api_key = OPENAI_API_KEY,  
  api_version = OPENAI_API_VERSION,
  azure_endpoint = OPENAI_API_ENDPOINT
)

def generate_embeddings_azure_openai(text = " ",model=EMBEDDING_MODEL_DEPLOYEMENT_NAME):
    response = client.embeddings.create(
        input = text,
        model= model
    )
    return response.data[0].embedding




def call_gpt_model(model= GPT4O,
                                  messages= [],
                                  temperature=0.1,
                                  max_tokens = 700,
                                  stream = True,seed= 999):

    print("Using model :",model)

    response = client.chat.completions.create(model=model,
                                              messages=messages,
                                              temperature = temperature,
                                              max_tokens = max_tokens,
                                              stream= stream,seed=seed)

    return response
    
system_message_query_generation_for_retriver = """
You are a very good text analyzer.
You will be provided a chat history and a user question.
You task is generate a search query that will return the best answer from the knowledge base.
Try and generate a grammatical sentence for the search query.
Do NOT use quotes and avoid other search operators.
Do not include cited source filenames and document names such as info.txt or doc.pdf in the search query terms.
Do not include any text inside [] or <<>> in the search query terms.
"""


def generate_query_for_retriver(user_query = " ",messages = [],model= "gpt35turbo"):

    start = time.time()
    user_message = summary_prompt_template = """Chat History:
    {chat_history}

    Question:
    {question}

    Search query:"""

    user_message = user_message.format(chat_history=str(messages), question=user_query)

    chat_conversations_for_query_generation_for_retriver = [{"role" : "system", "content" : system_message_query_generation_for_retriver}]
    chat_conversations_for_query_generation_for_retriver.append({"role": "user", "content": user_message })

    response = call_gpt_model(messages = chat_conversations_for_query_generation_for_retriver,stream = False,model= model)
    response = response.choices[0].message.content
    print("Generated Query for Retriver in :", time.time()-start,'seconds.')
    print("Generated Query for Retriver is :",response)

    return response
    
    
class retrive_similiar_docs: 
    def __init__(self, query=" ", retrive_fields=["actual_content", "metadata"], vector_field="embedding"):
        if query:
            self.query = query
        self.search_client = SearchClient(
            AZURE_COGNITIVE_SEARCH_ENDPOINT, 
            AZURE_COGNITIVE_SEARCH_INDEX_NAME, 
            azure_credential
        )
        self.retrive_fields = retrive_fields
        self.vector_field = vector_field  # new addition

    def text_search(self, top=2):
        results = self.search_client.search(
            search_text=self.query,
            select=self.retrive_fields,
            top=top
        )
        return results

    def pure_vector_search(self, k=2, query_embedding=[]):
        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=self.vector_field)
        results = self.search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=self.retrive_fields
        )
        return results

    def hybrid_search(self, top=2, k=2, query_embedding=[]):
        vector_query = RawVectorQuery(vector=query_embedding, k=k, fields=self.vector_field)
        results = self.search_client.search(
            search_text=self.query,
            vector_queries=[vector_query],
            select=self.retrive_fields,
            top=top
        )
        return results




import time
start = time.time()


def get_similiar_content(user_query = " ",
                      search_type = "hybrid",top = 4, k =4):

    #print("Generating query for embedding...")
    #embedding_query = get_query_for_embedding(user_query=user_query)
    retrive_docs = retrive_similiar_docs(query = user_query, vector_field=VECTOR_FIELD_NAME)


    if search_type == "text":
        start = time.time()
        r = retrive_docs.text_search(top =top)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        #print("Retrived Docs are :",sources,"\n")

    if search_type == "vector":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)
        print("Generated embedding for search query in :", time.time()-start,'seconds.')

        start = time.time()
        r = retrive_docs.pure_vector_search(k=k, query_embedding = vector_of_search_query)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
       # print("Retrived Docs are :",sources,"\n")


    if search_type == "hybrid":
        start = time.time()
        vector_of_search_query = generate_embeddings_azure_openai(user_query)
        print("Generated embedding for search query in :", time.time()-start,'seconds.')

        start = time.time()
        r = retrive_docs.hybrid_search(top = top, k=k, query_embedding = vector_of_search_query)

        sources = []
        similiar_doc = []
        for doc in r:
            similiar_doc.append(doc["metadata"] + ": " + doc["actual_content"].replace("\n", "").replace("\r", ""))
            sources.append(doc["metadata"])
        similiar_docs = "\n".join(similiar_doc)
        #print("*"*100)
        print("Retrived similiar documents with text search in :", time.time()-start,'seconds.')
        #print("similiar_doc :", similiar_doc)
        print("Retrived Docs are :",sources,"\n")
        source = " ".join(sources)
        print("Retrived Docs are after concat:",sources,"\n")
        #print("similiar_doc :", similiar_doc)
        #print("*"*100)
    return similiar_docs,source
    


system_message = """

You are a chatbot designed to answer user queries strictly related to documents stored in the HCM Knowledgebase, with a focus on client-specific context.

Your behavior must follow these rules:

Client Identification and Targeted Retrieval

Always begin by identifying the client, organization, or entity mentioned in the user's question (e.g., "Garnet Health", "billings clinic" etc..).

Use this client name to narrow your retrieval scope to only those documents that are related to or mention the identified client.

If no client is mentioned, perform a general retrieval across all available documents.

Answering Guidelines

Answer only the specific question posed by the user.

Be concise and direct, without providing extra commentary or unsolicited elaboration.

Base your responses strictly on the content available in the provided document extracts.

No Fabrication or Assumption

If the information required to answer the question is not found in the document extracts, respond exactly with:

"I don't have information to answer that question."

Do not infer, assume, or fabricate answers under any circumstances.

Source Referencing

When document content is used in the response, insert two newlines followed by the filename in square brackets:

"\n\n[Source: {filename}]"

Only provide this citation if the answer directly uses content from a document. Omit if not applicable.

Handling Greetings and Politeness

If the user greets (e.g., "Hi", "Hello") or thanks you, respond politely and ask how you may assist further. Do not include a document citation in such cases.

Strict adherence to these rules is required at all times. Do not deviate.
    
"""

chat_conversations_global_message = [{"role" : "system", "content" : system_message}]


def generate_response_with_memory(user_query = " ",keep_messages = 10,new_conversation = False,model="ibcgpt35turbo",stream=False):


    query_for_retriver = generate_query_for_retriver(user_query=user_query,messages = st.session_state.memory_messages[-keep_messages:],model=model)
    
    similiar_docs,sources = get_similiar_content(query_for_retriver)
    user_content = user_query + " \nSOURCES:\n" + similiar_docs

    chat_conversations_to_send = chat_conversations_global_message + st.session_state.memory_messages[-keep_messages:] + [{"role":"user","content" : user_content}]
    
    response_from_model = call_gpt_model(messages = chat_conversations_to_send,model=model)
    
    #sources = " ".join(sources)
    return response_from_model,query_for_retriver,sources




# User input
if prompt := st.chat_input("Please type your query here.?"):
    # Display user message in chat message container
    st.chat_message("user",avatar = "🧑‍💻").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    user_query = prompt
    if use_memory:
        response,query_for_retriver,sources = generate_response_with_memory(user_query= user_query,stream=True,model=GPT4O)
    print("##"*100)

    with st.chat_message("assistant",avatar = "🤖"):
        message_placeholder = st.empty()
        full_response = " "
        for chunk in response:
            if len(chunk.choices) >0:
                if str(chunk.choices[0].delta.content) != "None": 
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    st.session_state.memory_messages.append({"role": "user", "content": user_query})
    st.session_state.memory_messages.append({"role": "assistant", "content": full_response})
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    #extra_info = "Using Model :"+model_to_use
    #extra_info = 'Using Model : %(model_to_use)s , Generated query for retriver is : %(query_for_retriver)s \n, Source docs are - %(sources)s' % {'model_to_use': model_to_use,"query_for_retriver":query_for_retriver,"sources":sources}
    
    
    #print("Extra Info :", extra_info)
    #message_placeholder = st.empty()
    #message_placeholder.markdown(extra_info)
    #st.session_state.messages.append({"role": "assistant", "content": extra_info})
