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
import pandas as pd
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

    st.sidebar.error("1️⃣ Select your knowledge base and model")
    container_name = st.sidebar.selectbox("Knowledge base choice: ", container_list)
    model_variable = st.sidebar.selectbox("Model choice: ", ["o4-mini","gpt-4","gpt-4o","llama-4-Scout"])

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
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False
    if "current_warnings" not in st.session_state:
        st.session_state.current_warnings = []

    # Check if we need to reload data (container or model changed)
    need_reload = (
    st.session_state.current_container != container_name or 
    st.session_state.current_model != model_variable
    ) and st.session_state.data_loaded

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
    blob_list_df=pd.DataFrame(blob_list)
    st.sidebar.dataframe(blob_list_df["Name"], use_container_width=True)

    # detect zero‐byte blobs
    blob_list_df=pd.DataFrame(blob_list)
    if not blob_list_df.empty and "Size" in blob_list_df.columns:
        zero_size_files = blob_list_df[blob_list_df["Size"] == 0]
        if not zero_size_files.empty:
            st.sidebar.error(f"⚠️ {len(zero_size_files)} empty files detected!")
            names = zero_size_files["Name"].tolist()
            md = "**Empty files:**\n\n" + "\n".join(f"- {n}" for n in names)
            st.sidebar.markdown(md)


    st.header("Start chatting with your documents 💬 📚")
    
    # Show current selections
    # st.info(f"Selected: **{container_name}** using **{model_variable}**")

    # Add button to start/restart indexing
    st.error("2️⃣ Please click 'Load & Index Documents' to load the knowledge base")
    start_indexing = st.button("🔄 Load & Index Documents", type="primary",use_container_width=True)
    if st.session_state.data_loaded:
            st.success(f"✅ Knowledge base loaded! ")

            # Display any validation warnings
            for warning in st.session_state.current_warnings:
                st.warning(warning)

            st.error("3️⃣ Start chatting")
    

    # Reset messages when switching knowledge base or model
    if need_reload or start_indexing:

        st.session_state.current_warnings = []  # Clear old warnings

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
    
    # def validate_documents(knowledge_docs):
    #     """Validate that documents have content and are not empty"""
    #     if not knowledge_docs:
    #         return False, "No documents found in the selected knowledge base."
        
    #     valid_docs = []
    #     empty_files = []
        
    #     for doc in knowledge_docs:
    #         # Check if document has actual text content
    #         if hasattr(doc, 'text') and doc.text and doc.text.strip():
    #             # Additional check for minimum content length
    #             if len(doc.text.strip()) > 10:  # At least 10 characters
    #                 valid_docs.append(doc)
    #             else:
    #                 filename = doc.metadata.get('file_name', 'Unknown file')
    #                 empty_files.append(filename)
    #         else:
    #             filename = doc.metadata.get('file_name', 'Unknown file')
    #             empty_files.append(filename)
        
    #     if not valid_docs:
    #         if empty_files:
    #             return False, f"All documents appear to be empty or have no readable content. Empty files: {', '.join(empty_files)}"
    #         else:
    #             return False, "No valid documents with readable content found."
        
    #     if empty_files:
    #         st.warning(f"⚠️ Some files were skipped (empty or unreadable): {', '.join(empty_files)}")
        
    #     return True, f"Successfully validated documents."

    def validate_documents(knowledge_docs):
        """Validate that documents have content and are not empty"""
        print(f"DEBUG: Total documents received: {len(knowledge_docs)}")
        
        if not knowledge_docs:
            return False, "No documents found in the selected knowledge base."
        
        valid_docs = []
        file_stats = {}  # Track stats per file
        
        for i, doc in enumerate(knowledge_docs):
            filename = doc.metadata.get('file_name', f'Unknown file {i}')
            
            # Initialize file stats if not exists
            if filename not in file_stats:
                file_stats[filename] = {
                    'total_chunks': 0,
                    'valid_chunks': 0,
                    'invalid_chunks': 0,
                    'total_text_length': 0
                }
            
            file_stats[filename]['total_chunks'] += 1
            
            # Check if document has actual text content
            has_text_attr = hasattr(doc, 'text')
            text_content = getattr(doc, 'text', None)
            text_length = len(text_content.strip()) if text_content else 0
            
            if has_text_attr and text_content and text_content.strip():
                if len(text_content.strip()) > 10:  # At least 10 characters
                    valid_docs.append(doc)
                    file_stats[filename]['valid_chunks'] += 1
                    file_stats[filename]['total_text_length'] += text_length
                    print(f"DEBUG: Document {i+1} ({filename}): VALID ✅ ({text_length} chars)")
                else:
                    file_stats[filename]['invalid_chunks'] += 1
                    print(f"DEBUG: Document {i+1} ({filename}): TOO SHORT ⚠️ ({text_length} chars)")
            else:
                file_stats[filename]['invalid_chunks'] += 1
                print(f"DEBUG: Document {i+1} ({filename}): EMPTY/NO TEXT ❌")
        
        # Analyze file-level results
        completely_empty_files = []
        files_with_some_valid_content = []
        
        for filename, stats in file_stats.items():
            print(f"DEBUG: File '{filename}' summary:")
            print(f"  - Total chunks: {stats['total_chunks']}")
            print(f"  - Valid chunks: {stats['valid_chunks']}")
            print(f"  - Invalid chunks: {stats['invalid_chunks']}")
            print(f"  - Total text length: {stats['total_text_length']}")
            
            if stats['valid_chunks'] == 0:
                # No valid chunks at all
                completely_empty_files.append(filename)
            elif stats['invalid_chunks'] > 0:
                # Some chunks are invalid but file has valid content
                files_with_some_valid_content.append(filename)
        
        print(f"DEBUG: Valid documents total: {len(valid_docs)}")
        print(f"DEBUG: Completely empty files: {completely_empty_files}")
        print(f"DEBUG: Files with some valid content: {files_with_some_valid_content}")
        
        if not valid_docs:
            if completely_empty_files:
                return False, f"All documents appear to be empty or have no readable content. Empty files: {', '.join(completely_empty_files)}"
            else:
                return False, "No valid documents with readable content found."
        
        # Only warn about completely empty files, not files with some invalid chunks
        if completely_empty_files:
            warning_msg = f"⚠️ Some files were completely empty or unreadable: {', '.join(completely_empty_files)}"
            st.session_state.current_warnings = [warning_msg]
            print(f"DEBUG: Warning stored: {warning_msg}")
        else:
            st.session_state.current_warnings = []
            print("DEBUG: No warnings to store - all files have some valid content")

        return True, f"Successfully validated {len(valid_docs)} document chunks from {len(file_stats)} files."

    def load_data(llm_model, container_name):
       # """Load data without caching to ensure fresh data per session"""
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
                # Validate documents before indexing
                is_valid, validation_message = validate_documents(knowledge_docs)
                if not is_valid:
                    st.error(f"❌ Document validation failed: {validation_message}")
                    st.error("Please upload valid documents with readable content to this knowledge base.")
                    st.stop()

                # Filter out empty documents
                knowledge_docs = [doc for doc in knowledge_docs if hasattr(doc, 'text') and doc.text and len(doc.text.strip()) > 10]
                st.success(f"✅ {validation_message}")

            with st.spinner(text="Loading and indexing the provided docs – hang tight! This should take a couple of minutes."):

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
            similarity_top_k=10,
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
    if need_reload or start_indexing:
        try:
            st.session_state.index, knowledge_docs = load_data(model_variable, container_name)
            
            # Additional check after indexing
            if st.session_state.index is None:
                st.error("❌ Failed to create search index. Please check your documents.")
                st.session_state.data_loaded = False
                st.stop()
            
            st.session_state.chat_engine, st.session_state.memory = create_chat_engine(
                model_variable, st.session_state.index, container_name
            )
            st.session_state.current_container = container_name
            st.session_state.current_model = model_variable
            st.session_state.data_loaded = True
            st.success("Documents loaded and indexed successfully!")
            st.rerun()  # Refresh to update the UI state
            
        except Exception as e:
            st.error(f"❌ Error loading documents: {str(e)}")
            st.error("This usually happens when documents are empty or corrupted. Please check your files.")
            st.session_state.data_loaded = False
            st.stop()
        
    if not st.session_state.data_loaded:
        st.stop()

    if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        logger.info(f"User asked: {prompt} from {container_name} Knowledge base")# record into the log container

    for message in st.session_state.messages: # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])


    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    print(f"DEBUG: About to call chat_engine.chat() with prompt: '{prompt}'")
                    print(f"DEBUG: Chat engine exists: {st.session_state.chat_engine is not None}")
                    print(f"DEBUG: Current model: {st.session_state.current_model}")
                    print(f"DEBUG: Current container: {st.session_state.current_container}")
                    
                    response = st.session_state.chat_engine.chat(prompt)
                    
                    print(f"DEBUG: Response object type: {type(response)}")
                    print(f"DEBUG: Response object exists: {response is not None}")
                    
                    if response:
                        print(f"DEBUG: Response has 'response' attribute: {hasattr(response, 'response')}")
                        if hasattr(response, 'response'):
                            response_text = response.response
                            print(f"DEBUG: Response text type: {type(response_text)}")
                            print(f"DEBUG: Response text length: {len(response_text) if response_text else 0}")
                            print(f"DEBUG: Response text (first 200 chars): {repr(response_text[:200]) if response_text else 'None'}")
                            
                            if response_text and response_text.strip():
                                st.write(response_text)
                                message = {"role": "assistant", "content": response_text}
                                st.session_state.messages.append(message)
                                print(f"DEBUG: Successfully added response to messages")
                            else:
                                st.error("❌ Response is empty or contains only whitespace")
                                print("DEBUG: Response text is empty or whitespace only")
                        else:
                            print(f"DEBUG: Response object attributes: {dir(response)}")
                            st.error("❌ Response object doesn't have 'response' attribute")
                    
                    if not response or not hasattr(response, 'response') or not response.response:
                        st.error("❌ No response generated. The knowledge base might be empty or corrupted.")
                        print("DEBUG: Failed response validation checks")
                        
                except AssertionError as e:
                    print(f"DEBUG: AssertionError: {str(e)}")
                    st.error("❌ Error: The knowledge base appears to be empty or has no searchable content.")
                    st.error("Please upload documents with readable text content.")
                    st.stop()
                except Exception as e:
                    print(f"DEBUG: Exception during chat: {type(e).__name__}: {str(e)}")
                    import traceback
                    print(f"DEBUG: Full traceback:\n{traceback.format_exc()}")
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.stop()
