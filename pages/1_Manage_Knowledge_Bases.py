import streamlit as st

import openai
import llama_index
from llama_index.llms.openai import OpenAI
try:
  from llama_index import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
except ImportError:
  from llama_index.core import VectorStoreIndex, ServiceContext, Document, SimpleDirectoryReader
import time
from azure.storage.blob import BlobServiceClient
from io import BytesIO
from datetime import datetime

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import TokenTextSplitter
from helpers.azhelpers import upload_to_azure_storage, list_all_containers, list_all_files, delete_all_files, create_new_container

import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchField, SearchFieldDataType,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    AzureOpenAIVectorizer,
    SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields,
    SearchIndexerDataSourceConnection, SearchIndexerDataContainer,
    SearchIndexerSkillset, SplitSkill, InputFieldMappingEntry, OutputFieldMappingEntry,
    SearchIndexer, IndexingParameters, IndexingSchedule
)
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery, HybridSearch

from azure.search.documents.indexes.models import (
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile
)
import os
from dotenv import load_dotenv

from azure.search.documents.indexes.models import SearchIndexerSkill, SearchIndexerSkillset

# ── NEW/UPDATED: Azure AI Search + AOAI setup ──────────────────────────────────
import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient, SearchIndexerClient
from azure.search.documents.indexes.models import (
    # Index
    SearchIndex, SearchField, SearchFieldDataType, SearchableField, SimpleField,
    VectorSearch, HnswAlgorithmConfiguration, VectorSearchProfile,
    AzureOpenAIVectorizer, AzureOpenAIVectorizerParameters,
    SemanticSearch, SemanticConfiguration, SemanticPrioritizedFields,

    # Data source
    SearchIndexerDataSourceConnection, SearchIndexerDataContainer,

    # Skills
    SearchIndexerSkillset, SplitSkill, AzureOpenAIEmbeddingSkill,
    InputFieldMappingEntry, OutputFieldMappingEntry,

    # Index projections (chunk-per-row)
    SearchIndexerIndexProjection, SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters, IndexProjectionMode,

    # Indexer
    SearchIndexer, IndexingParameters, IndexingSchedule
)
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery, HybridSearch
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from azure.search.documents.indexes.models import SemanticField

from azure.search.documents.indexes.models import (
    SearchIndexerSkillset, SplitSkill, InputFieldMappingEntry, OutputFieldMappingEntry,
    SearchIndexerIndexProjection, SearchIndexerIndexProjectionSelector,
    SearchIndexerIndexProjectionsParameters, IndexProjectionMode,
    SearchIndexer, IndexingParameters
)
from azure.search.documents.indexes.models import AzureOpenAIEmbeddingSkill, SearchIndexerSkill
from azure.core.exceptions import HttpResponseError
import requests, json



load_dotenv() 
password_unicef =os.environ["APP_PASSWORD"]




def delete_kb_resources(container: str, *, drop_index: bool = True) -> None:
    """Delete the Search artifacts for a KB: indexer → skillset → data source → (index)."""
    names = kb_names(container)
    # Try to stop/clear indexer state (non-fatal if it fails)
    try:
        indexer_client.reset_indexer(names["indexer"])
    except Exception:
        pass

    # Delete indexer
    try:
        indexer_client.delete_indexer(names["indexer"])
    except ResourceNotFoundError:
        pass

    # Delete skillset
    try:
        indexer_client.delete_skillset(names["skillset"])
    except ResourceNotFoundError:
        pass

    # Delete data source
    try:
        indexer_client.delete_data_source_connection(names["datasource"])
    except ResourceNotFoundError:
        pass

    # Finally, delete the index itself
    if drop_index:
        try:
            index_client.delete_index(names["index"])
        except ResourceNotFoundError:
            pass

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
    
    # add genai- prefix
    name = f"genai-{name}"

    # Ensure minimum length of 3 characters
    if len(name) < 3:
        name = name + 'kb'  # Add 'kb' for knowledge base
    
    # Ensure maximum length of 63 characters
    if len(name) > 63:
        name = name[:63].rstrip('-')
    
    return name


def get_search_client_for_index(index_name: str) -> SearchClient:
    key = os.getenv("SEARCH_QUERY_KEY") or SEARCH_ADMIN_KEY  # query key in prod
    return SearchClient(SEARCH_ENDPOINT, index_name, AzureKeyCredential(key))

def render_indexer_status(container: str):
    """One-shot status panel (non-blocking). Call anywhere in UI."""
    names = kb_names(container)
    try:
        st.caption(f"Index: **{names['index']}**  •  Indexer: **{names['indexer']}**")
        s = indexer_client.get_indexer_status(names["indexer"])
        last = getattr(s, "last_result", None)
        last_status = getattr(last, "status", "n/a")
        last_end = getattr(last, "end_time", None)
        last_err = getattr(last, "error_message", "")
        st.write(f"Service: **{s.status}**  •  Last run: **{last_status}**"
                 + (f" at {last_end}" if last_end else "")
                 + (f"  •  {last_err}" if last_err else ""))
    except ResourceNotFoundError:
        st.warning("Indexer not found yet.")

def wait_until_index_ready_enhanced(container: str, timeout_sec: int = 120, poll_sec: int = 3) -> tuple[bool, str]:
    """
    Enhanced version with better feedback and longer timeout for large documents.
    Returns (success: bool, message: str)
    """
    names = kb_names(container)
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    start_time = time.time()  # Use time.time() with proper module import
    attempts = 0
    max_attempts = timeout_sec // poll_sec
    
    try:
        while attempts < max_attempts:
            elapsed = time.time() - start_time  # Use time.time() with proper module import
            progress = min(elapsed / timeout_sec, 0.95)  # Cap at 95% until completion
            progress_bar.progress(progress)
            
            try:
                s = indexer_client.get_indexer_status(names["indexer"])
                service_status = s.status
                last = getattr(s, "last_result", None)
                last_status = getattr(last, "status", None)
                last_err = getattr(last, "error_message", "")
                
                # Update status text
                if last_status == "inProgress":
                    status_text.info(f"🔄 Indexing in progress... ({elapsed:.0f}s elapsed)")
                elif last_status == "success":
                    progress_bar.progress(1.0)
                    status_text.empty()
                    return True, "Indexing completed successfully!"
                elif last_status in {"error", "transientFailure"}:
                    progress_bar.empty()
                    status_text.empty()
                    return False, f"Indexing failed: {last_err or 'Unknown error'}"
                else:
                    status_text.info(f"⏳ Preparing indexer... ({elapsed:.0f}s elapsed)")
                    
            except ResourceNotFoundError:
                status_text.warning("⚠️ Indexer not found, creating resources...")
            except Exception as e:
                status_text.error(f"Error checking status: {str(e)}")
            
            time.sleep(poll_sec)  # Use time.sleep() with proper module import
            attempts += 1
            
    finally:
        # Clean up UI elements
        progress_bar.empty()
        status_text.empty()
    
    # Timeout reached
    return False, f"Indexing is still in progress after {timeout_sec}s. Check Azure portal for completion."

def show_ready_badge_and_count(container: str):
    """Small success banner + chunk count once ready."""
    idx = kb_names(container)["index"]
    try:
        count = get_search_client_for_index(idx).get_document_count()
    except Exception:
        count = None
    if count is None:
        st.success(f"Index **{idx}** is ready.")
    else:
        st.success(f"Index **{idx}** is ready • **{count}** chunks indexed.")
    st.toast(f"“{container}” is ready to query ✅")

# for index in Azure Search 

SEARCH_ENDPOINT = os.environ.get("SEARCH_ENDPOINT").rstrip("/")
SEARCH_ADMIN_KEY = os.environ.get("SEARCH_ADMIN_KEY")
BLOB_CONNECTION_STRING =  os.environ.get("BLOB_CONNECTION_STRING")

AZURE_OPENAI_ENDPOINT =  os.environ.get("AZURE_OPENAI_ENDPOINT")  # https://<name>.openai.azure.com
AZURE_OPENAI_API_KEY =  os.environ.get("AZURE_OPENAI_API_KEY")    # omit if using MI/RBAC
AZURE_EMBEDDING_DEPLOYMENT =  os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
SEARCH_API_VERSION = os.getenv("AZURE_SEARCH_API_VERSION", "2025-05-01-preview")


if not SEARCH_ENDPOINT.startswith("https://"):
    raise RuntimeError("SEARCH_ENDPOINT must start with https://<service>.search.windows.net")

EMBED_DIM = 1536 if "3-small" in AZURE_EMBEDDING_DEPLOYMENT else 3072

index_client  = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_ADMIN_KEY))
indexer_client = SearchIndexerClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_ADMIN_KEY))


def put_skillset_raw(*, skillset_name: str, index_name: str):
    endpoint = os.environ["SEARCH_ENDPOINT"].rstrip("/")
    admin_key = os.environ["SEARCH_ADMIN_KEY"]

    aoai = os.environ["AZURE_OPENAI_ENDPOINT"].rstrip("/")
    if "openai.azure.com" not in aoai:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT must be like https://<name>.openai.azure.com")

    dep  = os.environ["AZURE_EMBEDDING_DEPLOYMENT"]
    dims = 1536 if "3-small" in dep else 3072

    url = f"{endpoint}/skillsets/{skillset_name}?api-version={SEARCH_API_VERSION}"
    headers = {"Content-Type": "application/json", "api-key": admin_key}

    split_skill = {
        "@odata.type": "#Microsoft.Skills.Text.SplitSkill",
        "name": "split",
        "description": "Split document text into ~1k-token pages",
        "context": "/document",
        "textSplitMode": "pages",
        "maximumPageLength": 512,
        "pageOverlapLength": 120,
        "inputs": [{"name": "text", "source": "/document/content"}],
        "outputs": [{"name": "textItems", "targetName": "pages"}]
    }

    embed_skill = {
        "@odata.type": "#Microsoft.Skills.Text.AzureOpenAIEmbeddingSkill",
        "name": "embed",
        "description": "Embed each page with Azure OpenAI",
        "context": "/document/pages/*",
        # IMPORTANT: use resourceUri + apiKey on this API version
        "resourceUri": aoai,
        "deploymentId": dep,
        "modelName": dep,
        "dimensions": dims,
        "inputs": [{"name": "text", "source": "/document/pages/*"}],
        "outputs": [{"name": "embedding", "targetName": "page_vector"}]
    }
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if api_key:
        embed_skill["apiKey"] = api_key  # not httpHeaders

    payload = {
        "name": skillset_name,
        "skills": [split_skill, embed_skill],
        "indexProjections": {
            "selectors": [{
                "targetIndexName": index_name,
                "parentKeyFieldName": "parent_id",
                "sourceContext": "/document/pages/*",
                "mappings": [
                    {"name": "content",       "source": "/document/pages/*"},
                    {"name": "contentVector", "source": "/document/pages/*/page_vector"},
                    {"name": "title",         "source": "/document/metadata_storage_name"},
                    {"name": "container",     "source": "/document/metadata_storage_container"},
                    {"name": "filepath",      "source": "/document/metadata_storage_path"}
                ]
            }],
            "parameters": {"projectionMode": "skipIndexingParentDocuments"}
        }
    }

    r = requests.put(url, headers=headers, data=json.dumps(payload))
    if not r.ok:
        raise RuntimeError(f"Skillset upsert failed: {r.status_code} {r.text}")

def kb_names(container: str):
    # s = sanitize_container_name(container)
    s = container
    return {
        "index":      f"kb-{s}",
        "datasource": f"ds-{s}",
        "skillset":   f"ss-{s}",
        "indexer":    f"ixr-{s}",
    }




def ensure_kb_index(container: str):
    names = kb_names(container)
    index_name = names["index"]

    # Vector search: HNSW + Azure OpenAI vectorizer (query-time embeddings)
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="aoai-hnsw", algorithm_configuration_name="hnsw", vectorizer_name="aoai")],
        vectorizers=[AzureOpenAIVectorizer(
            vectorizer_name="aoai",
            parameters=AzureOpenAIVectorizerParameters(
                resource_url=AZURE_OPENAI_ENDPOINT.rstrip("/"),
                deployment_name=AZURE_EMBEDDING_DEPLOYMENT,
                model_name=AZURE_EMBEDDING_DEPLOYMENT,
                api_key=AZURE_OPENAI_API_KEY or None,
            )
        )]
    )

    # 🔧 IMPORTANT: key must have keyword analyzer for index projections
    fields = [
        SearchField(  # <- changed from SimpleField to SearchField + keyword analyzer
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            analyzer_name="keyword",
            filterable=True,
        ),
        SimpleField(name="parent_id", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title", type=SearchFieldDataType.String, filterable=True, sortable=True, retrievable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, retrievable=True),
        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            vector_search_dimensions=EMBED_DIM,
            vector_search_profile_name="aoai-hnsw",
        ),
        SimpleField(name="container", type=SearchFieldDataType.String, filterable=True, facetable=True, sortable=True),
        SimpleField(name="filepath", type=SearchFieldDataType.String, filterable=True, sortable=True),
        SimpleField(name="page", type=SearchFieldDataType.Int32, filterable=True, sortable=True),
    ]

    semantic = SemanticSearch(
        configurations=[SemanticConfiguration(
            name="default",
            prioritized_fields=SemanticPrioritizedFields(
                title_field=SemanticField(field_name="title"),
                content_fields=[SemanticField(field_name="content")]
            )
        )]
    )

    idx = SearchIndex(name=index_name, fields=fields, vector_search=vector_search, semantic_search=semantic)

    try:
        # try update/create (works if index doesn't exist yet)
        index_client.create_or_update_index(idx)
    except HttpResponseError as e:
        # If schema conflicts (e.g., key field analyzer mismatch), drop and recreate
        try:
            index_client.delete_index(index_name)
        except Exception:
            pass
        index_client.create_index(idx)




def ensure_kb_pipeline(container: str):
    names = kb_names(container)

    # 1) Data source → this container only
    conn_str = os.getenv("SEARCH_BLOB_CONNECTION_STRING")
    if not conn_str:
        raise RuntimeError("SEARCH_BLOB_CONNECTION_STRING is missing.")
    ds = SearchIndexerDataSourceConnection(
        name=names["datasource"],
        type="azureblob",
        connection_string=conn_str,
        container=SearchIndexerDataContainer(name=container),  # use the arg
    )
    indexer_client.create_or_update_data_source_connection(ds)

    # 2) Skillset via RAW REST (bypasses SDK serialization issues)
    put_skillset_raw(skillset_name=names["skillset"], index_name=names["index"])

    # 3) Indexer ties it together (SDK is fine here)
    indexer = SearchIndexer(
        name=names["indexer"],
        data_source_name=names["datasource"],
        target_index_name=names["index"],
        skillset_name=names["skillset"],
        parameters=IndexingParameters(configuration={
            "dataToExtract": "contentAndMetadata",
            "parsingMode": "default",
            "failOnUnsupportedContentType": False
        }),
        # schedule=IndexingSchedule(interval="PT15M")  # optional
    )
    indexer_client.create_or_update_indexer(indexer)

def ensure_kb_resources(container: str):
    """Idempotently create/update index + pipeline for this KB."""
    ensure_kb_index(container)
    ensure_kb_pipeline(container)

def run_kb_indexer(container: str):
    """Trigger ingestion for this KB now."""
    names = kb_names(container)
    indexer_client.run_indexer(names["indexer"])

def show_indexer_status(container: str):
    names = kb_names(container)
    try:
        status = indexer_client.get_indexer_status(names["indexer"])
        st.caption(f"Indexer status: {status.status}. Last result: "
                   f"{getattr(status.last_result, 'status', 'n/a')} "
                   f"{getattr(status.last_result, 'error_message', '')}")
    except ResourceNotFoundError:
        st.caption("Indexer not found yet.")

password_input = st.text_input("Enter a password", type="password")

if password_input==password_unicef:


    with st.expander("Create a new Knowledge Base", expanded=False):
        new_container_name = st.text_input("Name your new Knowledge Base")
        create_container = st.button("Create", type='primary')
        if create_container:
            if new_container_name.strip():
                sanitized_container_name = sanitize_container_name(new_container_name)
                create_result = create_new_container(sanitized_container_name)

                # 🔹 Ensure Search index + pipeline, then run once
                ensure_kb_resources(sanitized_container_name)  # Use sanitized_container_name instead
                run_kb_indexer(sanitized_container_name)       # Use sanitized_container_name instead

                st.success(f"Created KB and Search pipeline for: {sanitized_container_name}")
                if sanitized_container_name != new_container_name.lower():
                    st.info(f"Note: sanitized to '{sanitized_container_name}'.")
                container_name = sanitized_container_name  # Use sanitized_container_name instead
            else:
                st.error("Please enter a valid container name.")


    left, right = st.columns(2)

    with left:
        container_name = st.selectbox("Manage this Knowledge Base", list_all_containers())
        delete_container = st.button(f"Delete all files in {container_name}", type='primary')
        if delete_container:
                with st.spinner(f"Deleting all resources for {container_name}..."):
                    # 1) Wipe blob files (your existing helper)
                    delete_all_files(container_name)

                    # 2) Tear down Search artifacts (indexer, skillset, data source, index)
                    delete_kb_resources(container_name, drop_index=True)
                
                st.success(f"✅ Deleted KB '{container_name}': files removed and index dropped.")

    with right:
        file_list = st.container()
        write_file_list()
        
        

    uploaded_files = st.file_uploader(f"Add files to {container_name}",
                                    type=["pdf", "docx"], accept_multiple_files=True)
    upload_confirm = st.button("Upload now")


    if upload_confirm and uploaded_files:
        
        # Upload files with progress
        upload_progress = st.progress(0)
        upload_status = st.empty()
        
        total_files = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files):
            upload_status.info(f"📁 Uploading {uploaded_file.name}...")
            upload_to_azure_storage(uploaded_file, container_name)
            upload_progress.progress((i + 1) / total_files)
        
        upload_status.success(f"✅ Uploaded {total_files} file(s) to {container_name}")
        upload_progress.empty()
        
        # Refresh file list
        write_file_list()

        # 🔹 Make sure resources exist (safe/idempotent), then reindex
        with st.spinner("Setting up search resources..."):
            ensure_kb_resources(container_name)
            run_kb_indexer(container_name)

        # 🔹 Enhanced indexing status with loading indicators
        st.info("🚀 Starting document indexing process...")
        
        # Wait for indexing to complete with enhanced feedback
        success, message = wait_until_index_ready_enhanced(container_name, timeout_sec=180)
        
        if success:
            st.success(f"🎉 {message}")
            show_ready_badge_and_count(container_name)
            st.balloons()  # Celebration animation!
        else:
            st.warning(f"⚠️ {message}")
            st.info("💡 You can continue using the app. Indexing will complete in the background.")
            
            # Show current status for reference
            with st.expander("Current Indexer Status"):
                render_indexer_status(container_name)

    elif upload_confirm and not uploaded_files:
        st.warning("⚠️ Please select files to upload first.")

else:
    st.error("Please enter the correct password to access the application.")
