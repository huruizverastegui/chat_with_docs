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
from helpers.azhelpers import upload_to_azure_storage, list_all_containers, list_all_files, delete_all_files, create_new_container

def write_file_list():
    blob_list = list_all_files(container_name)
    file_list.empty()
    with file_list:
        st.write(f"Files in {container_name}:")
        st.dataframe(blob_list, use_container_width=True)
    return

def sanitize_container_name(name):
    """
    Sanitize container name to comply with Azure Blob Storage naming rules:
    - Only lowercase letters, numbers, and hyphens
    - Must start and end with letter or number
    - Length between 3-63 characters
    - No consecutive hyphens
    """
    import re
    
    # Convert to lowercase
    name = name.lower()
    
    # Replace spaces and underscores with hyphens
    name = re.sub(r'[_\s]+', '-', name)
    
    # Remove any characters that aren't lowercase letters, numbers, or hyphens
    name = re.sub(r'[^a-z0-9-]', '', name)
    
    # Remove consecutive hyphens
    name = re.sub(r'-+', '-', name)
    
    # Remove leading/trailing hyphens
    name = name.strip('-')
    
    # Ensure minimum length of 3 characters
    if len(name) < 3:
        name = name + 'kb'  # Add 'kb' for knowledge base
    
    # Ensure maximum length of 63 characters
    if len(name) > 63:
        name = name[:63].rstrip('-')
    
    return name



with st.expander("Create a new Knowledge Base", expanded=False):
    new_container_name = st.text_input("Name your new Knowledge Base")
    create_container = st.button("Create", type='primary')
    if create_container:
        if new_container_name.strip():
            # Sanitize container name according to Azure rules
            sanitized_container_name = sanitize_container_name(new_container_name)
            created_container_name = create_new_container(sanitized_container_name)
            st.success(f"Created new Knowledge Base: {sanitized_container_name}")
            if sanitized_container_name != new_container_name.lower():
                st.info(f"Note: Container name was sanitized from '{new_container_name}' to '{sanitized_container_name}' to comply with Azure naming rules.")
            container_name = created_container_name
        else:
            st.error("Please enter a valid container name.")


left,right = st.columns(2)
with left:
   container_name = st.selectbox("Manage this Knowledge Base", list_all_containers())
   delete_container = st.button(f"Delete all files in {container_name}",type='primary')
   if delete_container:
         delete_all_files(container_name)
         st.success(f"Deleted all files in {container_name}")
with right:
    file_list = st.container()
    write_file_list()
        
        

uploaded_files = st.file_uploader(f"Add files to {container_name}", type=["pdf", "docx"], accept_multiple_files=True)
upload_confirm = st.button("Upload now")
if upload_confirm:
    for uploaded_file in uploaded_files:
        upload_to_azure_storage(uploaded_file, container_name)
        st.success(f"Uploaded {uploaded_file.name} to {container_name}")
    write_file_list()
