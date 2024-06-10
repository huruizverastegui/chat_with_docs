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

with st.expander("Create a new Knowledge Base", expanded=False):
    new_container_name = st.text_input("Name your new Knowledge Base")
    create_container = st.button("Create", type='primary')
    if create_container:
        created_container_name = create_new_container(new_container_name)
        st.success(f"Created new Knowledge Base: {new_container_name}")
        container_name = created_container_name


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