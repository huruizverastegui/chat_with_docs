import streamlit as st
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
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import TokenTextSplitter
from helpers.azhelpers import (
    upload_to_azure_storage,
    list_all_containers,
    list_all_files,
    Logger,
)

import os
from dotenv import load_dotenv

load_dotenv()
logger = Logger().get_logger()
logger.info("App started")
password_unicef = os.environ["APP_PASSWORD"]
password_input = st.text_input("Enter a password", type="password")

if password_input == password_unicef:
    azure_storage_account_name = os.environ["AZURE_STORAGE_ACCOUNT_NAME"]
    azure_storage_account_key = os.environ["AZURE_STORAGE_ACCOUNT_KEY"]
    connection_string_blob = os.environ["CONNECTION_STRING_BLOB"]

    blob_service_client = BlobServiceClient.from_connection_string(
        f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}"
    )

    container_list = list_all_containers()
    container_list = [
        container for container in container_list if container.startswith("genai")
    ]
    container_name = st.sidebar.selectbox("Answering questions from", container_list)
    model_variable = st.sidebar.selectbox(
        "Powered by", ["gpt-4", "gpt-4o", "gpt-3.5-turbo"]
    )
    st.sidebar.write("Using these documents:")
    blob_list = list_all_files(container_name)
    st.sidebar.dataframe(blob_list, use_container_width=True)

    openai.api_key = os.environ["OPEN_AI_KEY"]
    st.header("Start chatting with your documents 💬 📚")

    if "messages" not in st.session_state.keys():  # Initialize the chat message history
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask me a question about the documents you uploaded!",
            }
        ]

    @st.cache_resource(show_spinner=True)
    def load_data(llm_model, container_name):
        with st.spinner(
            text="Loading and indexing the provided docs – hang tight! This should take a couple of minutes."
        ):

            from llama_index.readers.azstorage_blob import AzStorageBlobReader

            loader = AzStorageBlobReader(
                container_name=container_name,
                connection_string=connection_string_blob,
            )

            knowledge_docs = loader.load_data()
            for doc in knowledge_docs:
                doc.excluded_embed_metadata_keys = [
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                ]
                doc.excluded_llm_metadata_keys = [
                    "file_type",
                    "file_size",
                    "creation_date",
                    "last_modified_date",
                    "last_accessed_date",
                ]

            service_context = ServiceContext.from_defaults(
                llm=OpenAI(
                    model=llm_model,
                    temperature=0.5,
                    system_prompt=""" Answer in a bullet point manner, be precise and provide examples. 
                            Keep your answers based on facts – do not hallucinate features.
                            Answer with all related knowledge docs. Always reference between phrases the ones you use. If you skip one, you will be penalized.
                            Use the format [file_name - page_label] between sentences. Use the exact same "file_name" and "page_label" present in the knowledge_docs.
                            Example:
                            The CPD priorities for Myanmar are strenghtening public education systems [2017-PL10-Myanmar-CPD-ODS-EN.pdf - page 2]
                            """,
                )
            )

            index = VectorStoreIndex.from_documents(
                knowledge_docs, service_context=service_context
            )
            return index

    index = load_data(model_variable, container_name)
    st.success("Documents loaded and indexed successfully!")

    memory = ChatMemoryBuffer.from_defaults(token_limit=5000)

    @st.cache_resource()  #
    def define_chat_engine():
        chat_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=memory,
            system_prompt=(
                """ Answer in a bullet point manner, be precise and provide examples.
                            Keep your answers based on facts – do not hallucinate features.
                            Answer with all related knowledge docs. Always reference between phrases the ones you use. If you skip one, you will be penalized.
                            Use the format [file_name - page_label] between sentences. Use the exact same "file_name" and "page_label" present in the knowledge_docs.
                            Example:
                            The CPD priorities for Myanmar are strenghtening public education systems [2017-PL10-Myanmar-CPD-ODS-EN.pdf - page 2]
                            """
            ),
            llm=OpenAI(model=model_variable, temperature=0.5),
        )
        return chat_engine

    chat_engine = define_chat_engine()

    if prompt := st.chat_input(
        "Your question"
    ):  # Prompt for user input and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        logger.info(f"User asked: {prompt} from {container_name} Knowledge base")

    for message in st.session_state.messages:  # Display the prior chat messages
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # If last message is not from assistant, generate a new response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_engine.chat(prompt)
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(
                    message
                )  # Add response to message history
