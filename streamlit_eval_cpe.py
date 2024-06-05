
import streamlit as st
import openai
import llama_index
from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader

from azure.storage.blob import BlobServiceClient
from io import BytesIO

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter


######### 
## Upload files to Azure blob 


# Azure Storage Account details
azure_storage_account_name = st.secrets.azure_storage_account_name
azure_storage_account_key = st.secrets.azure_storage_account_key
container_name = st.secrets.container_name
connection_string_blob =st.secrets.connection_string_blob


# Function to upload file to Azure Storage
def upload_to_azure_storage(file):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.name)
    blob_client.upload_blob(file)

# Streamlit App

def list_all_containers():
    container_list = list()
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    containers = blob_service_client.list_containers()
    for container in containers:
        container_list.append(container.name)
    return container_list

with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        container_name = st.selectbox("Answering questions from", list_all_containers())
    with col2:
        model_variable = st.selectbox("Powered by", ["gpt-4", "gpt-4o", "gpt-3.5-turbo"])



# get the list of files in context to display them
st.header("Documents already uploaded")
if st.button("Update the list"):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    blob_list=blob_service_client.get_container_client('genai').list_blobs()
    
    blob_list_display = []   
    for blob in blob_list:
        blob_list_display.append(blob.name)
    for i in blob_list_display:
        st.write(i)

if st.button("Reset and delete all uploaded documents"):
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
    blob_list=blob_service_client.get_container_client('genai').list_blobs()
    
    for blob in blob_list:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob.name)
        blob_client.delete_blob()
    st.write('Documents all deleted')



# if 'blob_list_display' not in st.session_state:
#     st.session_state.blob_list_display = []   
# for blob in blob_list:
#     st.session_state.blob_list_display.append(blob.name)
# for i in st.session_state.blob_list_display:
#     st.write(i)

# 



#update

st.header("Select documents to upload")
uploaded_files = st.file_uploader("Choose a document", accept_multiple_files=True)

if uploaded_files is not None:
    # st.image(uploaded_file)

    # Upload the file to Azure Storage on button click
    if st.button("Upload the documents"):
        for uploaded_file in uploaded_files:
            upload_to_azure_storage(uploaded_file)
            st.success("File uploaded to Azure Storage!")
        

openai.api_key = st.secrets.openai_key
st.header("Start chatting with your documents 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the documents you uploaded!"}
    ]
                                                        


@st.cache_resource(show_spinner=True)
def load_data(llm_model):
    with st.spinner(text="Loading and indexing the provided docs – hang tight! This should take a couple of minutes."):

        from llama_index.readers.azstorage_blob import AzStorageBlobReader

        loader = AzStorageBlobReader(
            container_name="genai",
            connection_string=connection_string_blob,
        )

        knowledge_docs = loader.load_data()
        for doc in knowledge_docs: 
            doc.excluded_embed_metadata_keys=['file_type','file_size','creation_date', 'last_modified_date','last_accessed_date']
            doc.excluded_llm_metadata_keys=['file_type','file_size','creation_date', 'last_modified_date','last_accessed_date']

        service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm_model, temperature=0.5, 
                        system_prompt=""" Answer in a bullet point manner, be precise and provide examples. 
                        Keep your answers based on facts – do not hallucinate features.
                        Answer with all related knowledge docs. Always reference between phrases the ones you use. If you skip one, you will be penalized.
                        Use the format [file_name - page_label] between sentences. Use the exact same "file_name" and "page_label" present in the knowledge_docs.
                        Example:
                        The CPD priorities for Myanmar are strenghtening public education systems [2017-PL10-Myanmar-CPD-ODS-EN.pdf - page 2]
                        """


                        ))
    
        ## Add the transfornations
        # text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=128)
        # title_extractor = TitleExtractor(nodes=5)
        # qa_extractor = QuestionsAnsweredExtractor(questions=3)

        index = VectorStoreIndex.from_documents(knowledge_docs, service_context=service_context)
        return index

index = load_data(model_variable)
st.success("Documents loaded and indexed successfully!")

# add a streamlit button that will run load_data() function
if st.button("Load and index the new documents provided"):
    load_data.clear()
    index = load_data(model_variable)
    st.success("New documents loaded and indexed successfully!")

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
