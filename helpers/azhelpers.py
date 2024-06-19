
import openai
import llama_index
from llama_index.llms.openai import OpenAI

try:
    from llama_index import (
        VectorStoreIndex,
        ServiceContext,
        Document,
        SimpleDirectoryReader,
    )
except ImportError:
    from llama_index.core import (
        VectorStoreIndex,
        ServiceContext,
        Document,
        SimpleDirectoryReader,
    )

from azure.storage.blob import BlobServiceClient
from io import BytesIO

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
import streamlit as st
import logging
import os 
import dotenv
from dotenv import load_dotenv

load_dotenv()

azure_storage_account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
azure_storage_account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
connection_string_blob = os.environ["CONNECTION_STRING_BLOB"]
container_name = None
logging_container_name = os.environ["LOGGING_CONTAINER_NAME"]
blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")



class AzureBlobStorageHandler(logging.Handler):
    def __init__(self, connection_string, container_name, blob_name):
        super().__init__()
        self.connection_string = connection_string
        self.container_name = container_name
        self.blob_name = blob_name
        self.blob_service_client = blob_service_client
        self.container_client = self.blob_service_client.get_container_client(
            container_name
        )

    def emit(self, record):
        log_entry = self.format(record)
        self.append_log_to_blob(log_entry)

    def append_log_to_blob(self, log_entry):
        blob_client = self.container_client.get_blob_client(self.blob_name)
        try:
            blob_data = blob_client.download_blob().content_as_text()
        except:
            blob_data = ""
        updated_log_data = blob_data + "\n" + log_entry
        blob_client.upload_blob(updated_log_data, overwrite=True)


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class Logger(metaclass=Singleton):
    def __init__(self):
        self.logger = logging.getLogger("azureLogger")
        self.logger.setLevel(logging.DEBUG)
        azure_handler = AzureBlobStorageHandler(
            connection_string_blob, logging_container_name, "test.log"
        )
        azure_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        azure_handler.setFormatter(formatter)
        self.logger.addHandler(azure_handler)

    def get_logger(self):
        return self.logger


def list_all_containers():
    container_list = list()
    containers = blob_service_client.list_containers()
    for container in containers:
        container_list.append(container.name)
    return container_list


def list_all_files(container_name):
    blob_list = blob_service_client.get_container_client(container_name).list_blobs()
    blob_list_display = []
    for blob in blob_list:
        blob_list_display.append(blob.name)
    return blob_list_display


def upload_to_azure_storage(file):
    blob_client = blob_service_client.get_blob_client(
        container=container_name, blob=file.name
    )
    blob_client.upload_blob(file)
    return True


def delete_all_files(container_name):
    container_client = blob_service_client.get_container_client(container_name)
    blob_list = container_client.list_blobs()
    for blob in blob_list:
        container_client.delete_blob(blob.name)
    return True


def create_new_container(container_name):
    genai_container = f"genai-{container_name}"
    blob_service_client.create_container(genai_container)
    return True


