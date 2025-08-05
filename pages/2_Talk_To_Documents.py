import streamlit as st
import openai
import llama_index
from llama_index.llms.openai import OpenAI
from llama_index.llms.azure_openai import AzureOpenAI 

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

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SimpleNodeParser

from llama_index.core.node_parser import SentenceSplitter

from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding

from llama_index.core import Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import TokenTextSplitter
from helpers.azhelpers import upload_to_azure_storage, list_all_containers, list_all_files, Logger


import os 
from dotenv import load_dotenv

from llama_index.readers.azstorage_blob import AzStorageBlobReader
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

load_dotenv()

logger = Logger().get_logger()
logger.info("App started")

password_unicef =os.environ["APP_PASSWORD"]
password_input = st.text_input("Enter a password", type="password")

if password_input==password_unicef:
    azure_storage_account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    azure_storage_account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
    connection_string_blob = os.environ["CONNECTION_STRING_BLOB"]

    # Azure OpenAI Configuration
    azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
    azure_openai_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
    
    # Deployment names for your Azure OpenAI models
    gpt4o_deployment = os.environ.get("AZURE_GPT4O_DEPLOYMENT", "gpt-4o")
    gpt4o_mini_deployment = os.environ.get("AZURE_GPT4O_MINI_DEPLOYMENT", "gpt-4o-mini")
    gpt4_deployment = os.environ.get("AZURE_GPT4_DEPLOYMENT", "gpt-4")
    gpt35_deployment = os.environ.get("AZURE_GPT35_DEPLOYMENT", "gpt-35-turbo")
    o4_mini_deployment =  os.environ.get("AZURE_O4MINI_DEPLOYMENT", "o4-mini")
    embedding_deployment = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")

  
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")

    container_list = list_all_containers()
    container_list = [container for container in container_list if container.startswith("genai")]
    container_name = st.sidebar.selectbox("Answering questions from", container_list)
    model_variable = st.sidebar.selectbox("Powered by", ["o4-mini","gpt-4","gpt-4o","llama-4-Scout"])

    # IMPORTANT: Initialize session state for user isolation
    if "current_container" not in st.session_state:
        st.session_state.current_container = None
    if "current_model" not in st.session_state:
        st.session_state.current_model = None
    if "index" not in st.session_state:
        st.session_state.index = None
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    if "memory" not in st.session_state:
        st.session_state.memory = None

    # Check if we need to reload data (container or model changed)
    need_reload = (
        st.session_state.current_container != container_name or 
        st.session_state.current_model != model_variable or
        st.session_state.index is None
    )

    # Get the API parameters for the Llama models hosted on Azure 
    if model_variable == "llama3-8B":
        azure_api_base = os.environ["URL_AZURE_LLAMA3_8B"]
        azure_api_key = os.environ["KEY_AZURE_LLAMA3_8B"]
      
    elif model_variable == "llama3-70B":
        azure_api_base = os.environ["URL_AZURE_LLAMA3_70B"]
        azure_api_key = os.environ["KEY_AZURE_LLAMA3_70B"]

    elif model_variable == "llama3.1-70B":
        azure_api_base = os.environ["URL_AZURE_LLAMA3_1_70B"]
        azure_api_key = os.environ["KEY_AZURE_LLAMA3_1_70B"]

    elif model_variable == "llama-4-Scout":
        azure_api_base = os.environ["URL_AZURE_LLAMA4_SCOUT"]
        azure_api_key = os.environ["KEY_AZURE_LLAMA4_SCOUT"]

    st.sidebar.write("Using these documents:")
    blob_list = list_all_files(container_name)
    st.sidebar.dataframe(blob_list, use_container_width=True)

    st.header("Start chatting with your documents 💬 📚")

    # Reset messages when switching knowledge base or model
    if need_reload:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about the documents you uploaded!"}
        ]
    elif "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask me a question about the documents you uploaded!"}
        ]

    def get_azure_openai_deployment_name(model_name):
        # """Map model names to Azure OpenAI deployment names"""
        deployment_mapping = {
            "gpt-4o": gpt4o_deployment,
            "gpt-4o-mini": gpt4o_mini_deployment,
            "gpt-4": gpt4_deployment,
            "gpt-3.5-turbo": gpt35_deployment,
            "o4-mini": o4_mini_deployment ,
        }
        return deployment_mapping.get(model_name, model_name)
    
    # Helper function to check if model is o1 series
    def is_o_model(model_name):
        return model_name in ["o4-mini"]

    def load_data(llm_model, container_name):
        """Load data without caching to ensure fresh data per session"""
        with st.spinner(text="Loading and indexing the provided docs – hang tight! This should take a couple of minutes."):

            loader = AzStorageBlobReader(
                container_name=container_name,
                connection_string=connection_string_blob,
            )

            knowledge_docs = loader.load_data()

            # Clean document metadata to remove non-serializable Azure objects
            def clean_doc_metadata(doc):
                # Remove problematic Azure-specific metadata
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    # Only keep serializable metadata
                    if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                        cleaned_metadata[key] = value
                    elif hasattr(value, '__dict__'):
                        # Convert objects to string representation
                        cleaned_metadata[key] = str(value)
                
                doc.metadata = cleaned_metadata
                
                # Set exclusion keys for embedding and LLM
                doc.excluded_embed_metadata_keys = [
                    'file_type', 'file_size', 'creation_date', 'last_modified_date', 
                    'last_accessed_date', 'copy_properties', 'blob_type', 'content_settings'
                ]
                doc.excluded_llm_metadata_keys = [
                    'file_type', 'file_size', 'creation_date', 'last_modified_date', 
                    'last_accessed_date', 'copy_properties', 'blob_type', 'content_settings'
                ]
                return doc

            with ThreadPoolExecutor(max_workers=4) as executor:
                knowledge_docs = list(executor.map(clean_doc_metadata, knowledge_docs))

            # define the model to use depending on the llm_model provided 
                        
            # 3 scenarios: Azure OpenAI GPT models, Llama models hosted on Azure, or fallback
            if llm_model in ["llama3-8B", "llama3-70B","llama3.1-70B","llama-4-Scout"] :
                llm_chat=OpenAI( api_base = azure_api_base ,
                            api_key = azure_api_key , 
                            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "5000")) ,
                            temperature=0.1,
                            system_prompt=""" Answer in a bullet point manner, be precise and provide examples. 
                                    Keep your answers based on facts – do not hallucinate features.
                                    Answer with all related knowledge docs. Always reference between phrases the ones you use. If you skip one, you will be penalized.
                                    Use the format [file_name - page_label] between sentences. Use the exact same "file_name" and "page_label" present in the knowledge_docs.
                                    Example:
                                    The CPD priorities for Myanmar are strenghtening public education systems [2017-PL10-Myanmar-CPD-ODS-EN.pdf - page 2]
                                    """ )
                
            elif llm_model in ["gpt-4o-mini","gpt-4", "gpt-4o", "gpt-3.5-turbo","o4-mini"]:
                deployment_name = get_azure_openai_deployment_name(llm_model)

                if is_o_model(llm_model):
                    llm_chat = AzureOpenAI( 
                    engine=deployment_name,
                    azure_endpoint=azure_openai_endpoint,
                    api_key=azure_openai_api_key,
                    api_version=azure_openai_api_version,
                    # No temperature parameter for o models
                    # No system_prompt parameter for o models
                    )
                    
                else:
                    llm_chat = AzureOpenAI( 
                                engine=deployment_name,
                                azure_endpoint=azure_openai_endpoint,
                                api_key=azure_openai_api_key,
                                api_version=azure_openai_api_version,
                                temperature=0.1,
                                system_prompt=""" Answer in a bullet point manner, be precise and provide examples. 
                                        Keep your answers based on facts – do not hallucinate features.
                                        Answer with all related knowledge docs. Always reference between phrases the ones you use. If you skip one, you will be penalized.
                                        Use the format [file_name - page_label] between sentences. Use the exact same "file_name" and "page_label" present in the knowledge_docs.
                                        Example:
                                        The CPD priorities for Myanmar are strenghtening public education systems [2017-PL10-Myanmar-CPD-ODS-EN.pdf - page 2]
                                        """ )

            Settings.llm = llm_chat
            
            Settings.embed_model = AzureOpenAIEmbedding(
                model=embedding_deployment,
                azure_endpoint=azure_openai_endpoint,
                api_key=azure_openai_api_key,
                api_version=azure_openai_api_version,
                embed_batch_size=20
            )

            Settings.node_parser = SentenceSplitter(
                chunk_size=1024,  # Larger chunks = fewer API calls
                chunk_overlap=100,
                paragraph_separator="\n\n",
                secondary_chunking_regex="[.!?]+",
            )
        
            index = VectorStoreIndex.from_documents(knowledge_docs)
            return index, knowledge_docs

    def create_chat_engine(llm_model, index, container_name):
        """Create a new chat engine for the current session"""
        
        # Create fresh memory for this session
        memory = ChatMemoryBuffer.from_defaults(token_limit=5000)

        if llm_model in ["llama3-8B", "llama3-70B","llama3.1-70B","llama-4-Scout"] :
            llm_chat=OpenAI( api_base = azure_api_base ,
                        api_key = azure_api_key , 
                        max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4000")) ,
                        temperature=0.1)
            
        elif llm_model in ["gpt-4o-mini","gpt-4", "gpt-4o", "gpt-3.5-turbo","o4-mini"]:
                deployment_name = get_azure_openai_deployment_name(llm_model)
                llm_chat = AzureOpenAI( 
                            engine=deployment_name,
                            azure_endpoint=azure_openai_endpoint,
                            api_key=azure_openai_api_key,
                            api_version=azure_openai_api_version,
                            temperature=0.1)
        
        # elif llm_model in ["o4-mini"]:
                if is_o_model(llm_model): 
                    deployment_name = get_azure_openai_deployment_name(llm_model)
                    llm_chat = AzureOpenAI( 
                                engine=deployment_name,
                                azure_endpoint=azure_openai_endpoint,
                                api_key=azure_openai_api_key,
                                api_version=azure_openai_api_version,
                                temperature=1)

        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            system_prompt=(
                f""" Answer in a bullet point manner, be precise and provide examples.
                        Keep your answers based on facts – do not hallucinate features. You are based on {llm_model} 
                        Answer with all related knowledge docs from {container_name} . Always reference between phrases the ones you use. If you skip one, you will be penalized.
                        Use the format [file_name - page_label] between sentences. Use the exact same "file_name" and "page_label" present in the knowledge_docs.
                        Example:
                        The CPD priorities for Myanmar are strenghtening public education systems [2017-PL10-Myanmar-CPD-ODS-EN.pdf - page 2]
                        """ if not is_o_model(llm_model) else None  # o-series models don't support system_prompt in chat_engine
            ),
            llm=llm_chat
        )
        return chat_engine, memory

    # Load data only when needed (container or model changed)
    if need_reload:
        st.session_state.index, knowledge_docs = load_data(model_variable, container_name)
        st.session_state.chat_engine, st.session_state.memory = create_chat_engine(
            model_variable, st.session_state.index, container_name
        )
        st.session_state.current_container = container_name
        st.session_state.current_model = model_variable
        st.success("Documents loaded and indexed successfully!")
    
    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        logger.info(f"User asked: {prompt} from {container_name} Knowledge base")# record into the log container

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message) # Add response to message history
                logger.info(f"{model_variable} Model answered: {response} -- from {container_name} Knowledge base")
