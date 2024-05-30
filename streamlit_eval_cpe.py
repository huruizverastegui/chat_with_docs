

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



######### 
## Upload files to Azure blob 


# Azure Storage Account details
azure_storage_account_name = st.secrets.azure_storage_account_name
azure_storage_account_key = st.secrets.azure_storage_account_key
container_name = st.secrets.container_name
connection_string_blob =st.secrets.connection_string_blob

# Password checks to grant access only to UNICEF staff
st.title("Chat with your documents 📚 🗣️")
password_unicef =st.secrets.password_unicef
password_input = st.text_input("Enter a password", type="password")


    
if password_input==password_unicef:


    # Function to upload file to Azure Storage
    def upload_to_azure_storage(file):
        blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=file.name)
        blob_client.upload_blob(file)

    # Streamlit App


    st.header("Select the documents to upload")
    uploaded_files = st.file_uploader("Choose a file", accept_multiple_files=True)

    if uploaded_files is not None:
        # st.image(uploaded_file)

        # Upload the file to Azure Storage on button click
        if st.button("Upload the documents"):
            for uploaded_file in uploaded_files:
                upload_to_azure_storage(uploaded_file)
                st.success("File uploaded to Azure Storage!")


    #########



    ######### NEW FEATURES
    ## ADD a list of models to choose from 
    ## GPT 4, 4o, 3.5 turbo
    ## LLama 3 and 4
    #########
    ## return the list of documents loaded in context 

    #########
    ##WAIT UNTILL THE DOCUMENTS ARE LOADED 
    ## SHOW A SPINNER
    #########

    # choose model from a list 
    model_variable = st.selectbox("Choose a model", ["gpt-4o", "gpt-4", "gpt-3.5-turbo"])

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

            docs = loader.load_data()

            service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm_model, temperature=0.5, system_prompt="You are an expert on the UNICEF country evaluation process and your job is to summarize context documents to help UNICEF staff in writing reports. Answer in a bullet point manner, be precise and provide examples. It's extremely important that you source each one of the bullet points. Keep your answers based on facts – do not hallucinate features."))
            index = VectorStoreIndex.from_documents(docs, service_context=service_context)
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
