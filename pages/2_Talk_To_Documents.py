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

# ===== NEW IMPORTS FOR AZURE AI SEARCH INTEGRATION =====
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore
from llama_index.core import StorageContext
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery, HybridSearch
from azure.search.documents.indexes import SearchIndexClient
from azure.core.exceptions import ResourceNotFoundError

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery, HybridSearch
from azure.core.credentials import AzureKeyCredential

import os 
from dotenv import load_dotenv
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
import sys

import psutil
import threading
from datetime import datetime
import json
import logging

import re
from typing import Tuple, Optional, List, Dict
from pathlib import Path

class PerformanceLogger:
    def __init__(self, logger):
        self.logger = logger
        self.timers = {}
        self.metrics = {}
    
    def start_timer(self, operation):
        """Start timing an operation"""
        self.timers[operation] = time.time()
        self.logger.info(f"🟡 STARTED: {operation} at {datetime.now().strftime('%H:%M:%S')}")
    
    def end_timer(self, operation, additional_info=""):
        """End timing and log duration"""
        if operation in self.timers:
            duration = time.time() - self.timers[operation]
            self.logger.info(f"✅ COMPLETED: {operation} in {duration:.2f}s {additional_info}")
            self.metrics[operation] = duration
            del self.timers[operation]
            return duration
        return 0
    
    def log_system_resources(self, stage):
        """Log current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.logger.info(f"📊 RESOURCES at {stage}: CPU: {cpu_percent}%, Memory: {memory.percent}% ({memory.used//1024//1024}MB used)")
        except Exception as e:
            self.logger.info(f"⚠️ Could not get system resources: {e}")
    
    def log_document_stats(self, docs, stage):
        """Log document statistics"""
        if not docs:
            self.logger.info(f"📄 DOCS at {stage}: No documents")
            return
        
        total_chars = sum(len(doc.text) if hasattr(doc, 'text') and doc.text else 0 for doc in docs)
        file_counts = {}
        
        for doc in docs:
            filename = doc.metadata.get('file_name', 'Unknown')
            file_counts[filename] = file_counts.get(filename, 0) + 1
        
        self.logger.info(f"📄 DOCS at {stage}: {len(docs)} chunks, {total_chars:,} total chars")
        for filename, count in file_counts.items():
            self.logger.info(f"    - {filename}: {count} chunks")




load_dotenv()

logger = Logger().get_logger()

# Add console output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)

# Initialize performance logger
perf_logger = PerformanceLogger(logger)

logger.info("App started")

password_unicef =os.environ["APP_PASSWORD"]
password_input = st.text_input("Enter a password", type="password")


# def analyze_query_complexity(query: str, total_documents: int = 0, total_files: int = 0) -> Tuple[str, int]:
#     """
#     Analyze query complexity and return complexity level with recommended similarity_top_k
    
#     Args:
#         query: The user's query
#         total_documents: Total number of document chunks in the knowledge base
#         total_files: Total number of unique files in the knowledge base
    
#     Returns:
#         Tuple of (complexity_level, recommended_top_k)
#     """
#     query = query.strip().lower()
#     query_length = len(query)
#     word_count = len(query.split())
    
#     # Document corpus size multipliers for scaling top_k
#     def get_corpus_multiplier(docs: int, files: int) -> float:
#         """Calculate multiplier based on corpus size"""
#         if files <= 5:
#             return 0.7  # Small corpus, reduce top_k
#         elif files <= 15:
#             return 1.0  # Medium corpus, standard top_k
#         elif files <= 50:
#             return 1.3  # Large corpus, increase top_k
#         else:
#             return 1.6  # Very large corpus, significantly increase top_k
    
#     corpus_multiplier = get_corpus_multiplier(total_documents, total_files)
    
#     # Simple greetings and basic queries
#     simple_patterns = [
#         r'^(hi|hello|hey|good morning|good afternoon|good evening)$',
#         r'^(thanks?|thank you|ok|okay)$',
#         r'^(yes|no|sure|maybe)$',
#         r'^test$',
#         r'^\w{1,10}$'  # Single very short words
#     ]
    
#     # Check if it's a simple greeting/response
#     if any(re.match(pattern, query) for pattern in simple_patterns):
#         base_top_k = 3
#         adjusted_top_k = max(3, int(base_top_k * corpus_multiplier))
#         return "simple", adjusted_top_k
    
#     # Short factual questions (usually need few sources, but scale with corpus)
#     if query_length < 50 and word_count <= 8:
#         # Look for question words that suggest simple factual queries
#         simple_question_words = ['what', 'when', 'where', 'who', 'which', 'how many']
#         if any(word in query for word in simple_question_words):
#             base_top_k = 5
#             adjusted_top_k = max(5, int(base_top_k * corpus_multiplier))
#             return "short_factual", adjusted_top_k
        
#         base_top_k = 5
#         adjusted_top_k = max(5, int(base_top_k * corpus_multiplier))
#         return "short", adjusted_top_k
    
#     # Cross-document analysis indicators (these REALLY need more chunks with large corpus)
#     cross_doc_keywords = [
#         'across documents', 'all documents', 'compare documents', 'between documents',
#         'in each document', 'document by document', 'comprehensive', 'overall',
#         'summarize everything', 'all files', 'entire knowledge base'
#     ]
    
#     # Complex analysis requests
#     analysis_keywords = [
#         'analyze', 'compare', 'contrast', 'evaluate', 'summarize', 'summary',
#         'trends', 'patterns', 'detailed analysis', 'in-depth', 'thorough'
#     ]
    
#     has_cross_doc = any(keyword in query for keyword in cross_doc_keywords)
#     has_analysis = any(keyword in query for keyword in analysis_keywords)
    
#     if has_cross_doc or (has_analysis and total_files > 10):
#         # Cross-document queries need significantly more chunks
#         base_top_k = 18
#         # Extra multiplier for cross-document queries
#         cross_doc_multiplier = corpus_multiplier * 1.4
#         adjusted_top_k = max(12, min(25, int(base_top_k * cross_doc_multiplier)))
#         return "cross_document", adjusted_top_k
#     elif has_analysis:
#         base_top_k = 12
#         adjusted_top_k = max(8, int(base_top_k * corpus_multiplier))
#         return "complex", adjusted_top_k
    
#     # Medium complexity - typical questions
#     elif query_length < 150 and word_count <= 25:
#         base_top_k = 8
#         adjusted_top_k = max(6, int(base_top_k * corpus_multiplier))
#         return "medium", adjusted_top_k
    
#     # Very long or comprehensive requests
#     elif query_length > 200 or word_count > 35:
#         base_top_k = 15
#         adjusted_top_k = max(10, int(base_top_k * corpus_multiplier))
#         return "comprehensive", adjusted_top_k
    
#     # Default medium complexity
#     base_top_k = 8
#     adjusted_top_k = max(6, int(base_top_k * corpus_multiplier))
#     return "medium", adjusted_top_k


# def optimize_similarity_top_k(query: str, default_top_k: int, knowledge_docs=None, logger=None) -> int:
#     """
#     Dynamically adjust similarity_top_k based on query analysis and content volume
    
#     Args:
#         query: The user's query
#         default_top_k: The default similarity_top_k from session state
#         knowledge_docs: List of documents to analyze content volume (optional)
#         logger: Logger instance for detailed logging (optional)
        
#     Returns:
#         Optimized similarity_top_k value
#     """
#     # Get content statistics
#     if knowledge_docs:
#         total_chunks = len(knowledge_docs)
#         total_content_length = sum(len(doc.text) if hasattr(doc, 'text') and doc.text else 0 
#                                  for doc in knowledge_docs)
        
#         # Get unique file count for additional context
#         unique_files = len(set(doc.metadata.get('file_name', 'unknown') 
#                              for doc in knowledge_docs))
#     else:
#         total_chunks = 0
#         unique_files = 0
    
#     complexity, recommended_top_k = analyze_query_complexity(
#         query, total_chunks, unique_files
#     )
    
#     # Enhanced logging with content volume metrics if logger provided
#     if logger:
#         content_mb = (sum(len(doc.text) if hasattr(doc, 'text') and doc.text else 0 
#                          for doc in knowledge_docs) / (1024 * 1024)) if knowledge_docs else 0
#         logger.info(f"🔍 QUERY ANALYSIS: '{query[:50]}...'")
#         logger.info(f"   Complexity: {complexity}")
#         logger.info(f"   Content Volume: {total_chunks} chunks, {content_mb:.1f}MB total")
#         logger.info(f"   Files: {unique_files}")
#         logger.info(f"   Recommended top_k: {recommended_top_k} (user default: {default_top_k})")
    
#     # Decision logic: balance user preference with content-aware optimization
#     if complexity in ["simple"]:
#         # For simple queries, prioritize speed even with large corpus
#         optimized_top_k = min(recommended_top_k, default_top_k)
#     elif complexity in ["short", "short_factual"]:
#         # For short factual queries, respect content scaling but don't go overboard
#         if default_top_k < recommended_top_k * 0.7:
#             optimized_top_k = default_top_k  # User really wants minimal chunks
#         else:
#             optimized_top_k = recommended_top_k
#     elif complexity in ["cross_document", "comprehensive"]:
#         # For cross-document queries, may need more than user default for quality
#         optimized_top_k = max(recommended_top_k, default_top_k)
#     else:
#         # For medium/complex, balance optimization with user preference
#         if default_top_k < (recommended_top_k * 0.6):  # User wants much fewer
#             optimized_top_k = max(default_top_k, int(recommended_top_k * 0.7))  # Compromise
#         else:
#             optimized_top_k = recommended_top_k
    
#     # Ensure we don't exceed reasonable bounds
#     optimized_top_k = max(3, min(25, optimized_top_k))
    
#     if logger and optimized_top_k != default_top_k:
#         direction = "↗️ Increased" if optimized_top_k > default_top_k else "📉 Reduced"
#         percentage_change = ((optimized_top_k - default_top_k) / default_top_k) * 100
#         logger.info(f"{direction} similarity_top_k from {default_top_k} to {optimized_top_k} ({percentage_change:+.0f}%)")
        
#         # Explain the reasoning
#         if complexity == "cross_document":
#             logger.info(f"   Reason: Cross-document query needs broad coverage across {unique_files} files")
#         elif complexity == "simple" and optimized_top_k < default_top_k:
#             logger.info(f"   Reason: Simple query - prioritizing speed over exhaustive search")
#         elif total_chunks > 200:
#             logger.info(f"   Reason: Large corpus ({total_chunks} chunks) - scaling for better coverage")
    
#     return optimized_top_k


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

def search_azure_directly(container_name, query, top_k=5):
    """
    Search Azure AI Search directly without LlamaIndex
    Only needs embedding for the query, not documents
    """
    try:
        # Azure Search setup
        search_endpoint = os.environ.get("SEARCH_ENDPOINT")
        search_admin_key = os.environ.get("SEARCH_ADMIN_KEY")
        index_name = f"kb-{container_name.lower().replace('_', '-')}"
        
        logger.info(f"🔍 Searching Azure index: {index_name} with semantic ranking enabled")
        logger.info(f"📊 Search parameters: semantic_query='{query[:50]}...', semantic_config='default', hybrid_search=True")
        
        # Create search client
        search_client = SearchClient(
            search_endpoint, 
            index_name, 
            AzureKeyCredential(search_admin_key)
        )
        
        # Create vector query (this is where embedding is used - for the query only!)
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=top_k,
            fields="contentVector"  # Your vector field name in Azure Search
        )
        
        # Search with hybrid approach (vector + keyword) + semantic ranking
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            hybrid_search=HybridSearch(),
            semantic_query=query,
            semantic_configuration_name="default",
            query_language="en-us",
            top=top_k,
            include_total_count=True
        )
        
        # Extract results
        documents = []
        for result in results:
            documents.append({
                'content': result.get('content', ''),
                'title': result.get('title', ''),
                'filepath': result.get('filepath', ''),
                'score': result.get('@search.score', 0)
            })
        
        logger.info(f"✅ Found {len(documents)} relevant documents")
        return documents
        
    except Exception as e:
        logger.error(f"❌ Azure Search error: {str(e)}")
        return []

def search_azure_with_document_filter(container_name, query, selected_documents, total_chunks=70, min_chunks_per_doc=5, mmr_enabled=False, mmr_lambda=0.5):
    """
    Search Azure AI Search with optimized multi-document retrieval strategy
    Makes a single API call with filter for all selected documents
    Enforces minimum chunks per document
    """
    try:
        # Azure Search setup
        search_endpoint = os.environ.get("SEARCH_ENDPOINT")
        search_admin_key = os.environ.get("SEARCH_ADMIN_KEY")
        index_name = f"kb-{container_name.lower().replace('_', '-')}"
        
        logger.info(f"🔍 Optimized multi-document search with semantic ranking: {len(selected_documents)} documents, {total_chunks} total chunks, min {min_chunks_per_doc} per doc")
        
        # Create search client
        search_client = SearchClient(
            search_endpoint, 
            index_name, 
            AzureKeyCredential(search_admin_key)
        )
        
        # Calculate required chunks to ensure minimum per document
        required_chunks = max(total_chunks, len(selected_documents) * min_chunks_per_doc)
        
        # Create vector query for all selected documents
        vector_query = VectorizableTextQuery(
            text=query,
            k_nearest_neighbors=required_chunks,
            fields="contentVector"
        )
        
        # Build filter for all selected documents
        if len(selected_documents) == 1:
            # Single document filter
            doc_filter = f"title eq '{list(selected_documents)[0]}'"
        else:
            # Multiple documents filter using 'or'
            doc_filters = [f"title eq '{doc}'" for doc in selected_documents]
            doc_filter = " or ".join(doc_filters)
        
        # Single search call for all selected documents with semantic ranking
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            hybrid_search=HybridSearch(),
            semantic_query=query,
            semantic_configuration_name="default",
            query_language="en-us",
            filter=doc_filter,
            top=required_chunks,
            include_total_count=True
        )
        
        # Extract all results
        all_results = []
        for result in results:
            all_results.append({
                'content': result.get('content', ''),
                'title': result.get('title', ''),
                'filepath': result.get('filepath', ''),
                'score': result.get('@search.score', 0)
            })
        
        # Group results by document for analysis
        results_by_doc = {}
        for result in all_results:
            doc_name = result['title']
            if doc_name not in results_by_doc:
                results_by_doc[doc_name] = []
            results_by_doc[doc_name].append(result)
        
        # Check for documents with insufficient chunks and fetch more if needed
        final_results = []
        for doc_name in selected_documents:
            doc_results = results_by_doc.get(doc_name, [])
            
            if len(doc_results) < min_chunks_per_doc:
                # Fetch additional chunks for this document to meet minimum
                logger.info(f"📄 {doc_name}: Only {len(doc_results)} chunks, fetching more to meet minimum {min_chunks_per_doc}")
                st.write(f"📄 **{doc_name}**: Only {len(doc_results)} chunks, fetching more...")
                
                additional_chunks_needed = min_chunks_per_doc - len(doc_results)
                additional_results = search_client.search(
                    search_text=query,
                    vector_queries=[VectorizableTextQuery(
                        text=query,
                        k_nearest_neighbors=additional_chunks_needed,
                        fields="contentVector"
                    )],
                    hybrid_search=HybridSearch(),
                    semantic_query=query,
                    semantic_configuration_name="default",
                    query_language="en-us",
                    filter=f"title eq '{doc_name}'",
                    top=additional_chunks_needed,
                    include_total_count=True
                )
                
                # Add additional chunks
                for result in additional_results:
                    if result.get('title') == doc_name:
                        doc_results.append({
                            'content': result.get('content', ''),
                            'title': result.get('title', ''),
                            'filepath': result.get('filepath', ''),
                            'score': result.get('@search.score', 0)
                        })
            
            # Add all chunks for this document to final results
            final_results.extend(doc_results)
            
            # Log final count for this document
            logger.info(f"📄 {doc_name}: {len(doc_results)} chunks (minimum {min_chunks_per_doc} enforced)")
            st.write(f"📄 **{doc_name}**: {len(doc_results)} chunks found")
        
        logger.info(f"✅ Optimized search complete: {len(final_results)} total chunks from {len(selected_documents)} documents")
        
        # Apply MMR for diversity if enabled and we have enough chunks
        if mmr_enabled and len(final_results) > 10:
            logger.info(f"🔄 Applying MMR for diversity improvement (λ={mmr_lambda})")
            final_results = apply_mmr(final_results, query, target_count=len(final_results), lambda_param=mmr_lambda)
        
        return final_results
        
    except Exception as e:
        logger.error(f"❌ Optimized search error: {str(e)}")
        return []

def apply_mmr(chunks, query, target_count, lambda_param=0.5):
    """
    Apply Maximal Marginal Relevance (MMR) to select diverse chunks
    """
    if len(chunks) <= target_count:
        return chunks
    
    # Sort chunks by relevance score (descending)
    sorted_chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
    
    # Start with most relevant chunk
    selected = [sorted_chunks[0]]
    remaining = sorted_chunks[1:]
    
    logger.info(f"🔄 Applying MMR: {len(chunks)} chunks → {target_count} diverse chunks")
    
    while len(selected) < target_count and remaining:
        best_mmr_score = -1
        best_chunk = None
        best_index = -1
        
        for i, chunk in enumerate(remaining):
            # Relevance score (already normalized from search)
            relevance = chunk['score']
            
            # Calculate max similarity to already selected chunks
            max_similarity = 0
            for selected_chunk in selected:
                similarity = calculate_chunk_similarity(chunk, selected_chunk)
                max_similarity = max(max_similarity, similarity)
            
            # Calculate MMR score
            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
            
            if mmr_score > best_mmr_score:
                best_mmr_score = mmr_score
                best_chunk = chunk
                best_index = i
        
        # Add best chunk to selected
        selected.append(best_chunk)
        remaining.pop(best_index)
    
    logger.info(f"✅ MMR complete: Selected {len(selected)} diverse chunks")
    return selected

def calculate_chunk_similarity(chunk1, chunk2):
    """
    Calculate similarity between two chunks using content-based approach
    """
    try:
        # Simple keyword-based similarity (faster than embeddings)
        content1 = chunk1['content'].lower()
        content2 = chunk2['content'].lower()
        
        # Extract words (simple tokenization)
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        if union == 0:
            return 0.0
        
        similarity = intersection / union
        
        # Normalize to 0-1 range and apply some smoothing
        return min(similarity * 2, 1.0)  # Scale up similarity, cap at 1.0
        
    except Exception as e:
        logger.warning(f"⚠️ Error calculating chunk similarity: {str(e)}")
        return 0.0

def group_chunks_by_document(search_results, max_chunks_per_group=20):
    """
    Group retrieved chunks by combining documents until reaching max_chunks_per_group
    Prioritizes document narrative integrity while optimizing for chunk count
    Returns list of document groups, each containing up to max_chunks_per_group chunks
    """
    try:
        # Group chunks by document title first
        chunks_by_doc = {}
        for chunk in search_results:
            doc_name = chunk.get('title', 'Unknown')
            if doc_name not in chunks_by_doc:
                chunks_by_doc[doc_name] = []
            chunks_by_doc[doc_name].append(chunk)
        
        # Sort chunks within each document by relevance score (descending)
        for doc_name in chunks_by_doc:
            chunks_by_doc[doc_name] = sorted(chunks_by_doc[doc_name], key=lambda x: x.get('score', 0), reverse=True)
        
        # Create optimized document groups
        document_groups = []
        current_group_chunks = []
        current_group_docs = []
        current_group_count = 0
        
        # Process documents in order of total chunk count (largest first)
        sorted_docs = sorted(chunks_by_doc.items(), key=lambda x: len(x[1]), reverse=True)
        
        for doc_name, doc_chunks in sorted_docs:
            # If this document alone exceeds max_chunks_per_group, split it first
            if len(doc_chunks) > max_chunks_per_group:
                # Save current group if it has chunks
                if current_group_chunks:
                    group_name = f"Group {len(document_groups) + 1}: {', '.join(current_group_docs)}"
                    document_groups.append({
                        'document_name': group_name,
                        'chunks': current_group_chunks,
                        'chunk_count': current_group_count,
                        'documents_included': current_group_docs.copy()
                    })
                    current_group_chunks = []
                    current_group_docs = []
                    current_group_count = 0
                
                # Split large document into multiple groups
                for i in range(0, len(doc_chunks), max_chunks_per_group):
                    group_chunks = doc_chunks[i:i + max_chunks_per_group]
                    group_name = f"{doc_name} (Part {i//max_chunks_per_group + 1})"
                    document_groups.append({
                        'document_name': group_name,
                        'chunks': group_chunks,
                        'chunk_count': len(group_chunks),
                        'documents_included': [doc_name]
                    })
            else:
                # Check if adding this document would exceed the limit
                if current_group_count + len(doc_chunks) <= max_chunks_per_group:
                    # Add this document to current group
                    current_group_chunks.extend(doc_chunks)
                    current_group_docs.append(doc_name)
                    current_group_count += len(doc_chunks)
                else:
                    # Current group is full, save it and start a new one
                    if current_group_chunks:
                        group_name = f"Group {len(document_groups) + 1}: {', '.join(current_group_docs)}"
                        document_groups.append({
                            'document_name': group_name,
                            'chunks': current_group_chunks,
                            'chunk_count': current_group_count,
                            'documents_included': current_group_docs.copy()
                        })
                    
                    # Start new group with this document
                    current_group_chunks = doc_chunks.copy()
                    current_group_docs = [doc_name]
                    current_group_count = len(doc_chunks)
        
        # Don't forget the last group
        if current_group_chunks:
            group_name = f"Group {len(document_groups) + 1}: {', '.join(current_group_docs)}"
            document_groups.append({
                'document_name': group_name,
                'chunks': current_group_chunks,
                'chunk_count': current_group_count,
                'documents_included': current_group_docs.copy()
            })
        
        logger.info(f"📄 Document grouping complete: {len(document_groups)} groups from {len(chunks_by_doc)} documents")
        for group in document_groups:
            docs_str = ', '.join(group.get('documents_included', [group['document_name']]))
            logger.info(f"   - {group['document_name']}: {group['chunk_count']} chunks from {len(group.get('documents_included', [group['document_name']]))} document(s)")
        
        return document_groups
        
    except Exception as e:
        logger.error(f"❌ Error grouping chunks by document: {str(e)}")
        return []

def search_document_groups(container_name, query, selected_documents, total_chunks=70, min_chunks_per_doc=5, max_chunks_per_group=20, mmr_enabled=False):
    """
    Search Azure AI Search and group results by document
    Returns document groups ready for multi-prompt processing
    """
    try:
        # First, get all chunks using existing search function
        logger.info(f"🔍 Starting document-grouped search: {len(selected_documents)} documents, {total_chunks} total chunks")
        
        search_results = search_azure_with_document_filter(
            container_name, 
            query, 
            selected_documents,
            total_chunks,
            min_chunks_per_doc,
            mmr_enabled=mmr_enabled,
            mmr_lambda=0.5
        )
        
        if not search_results:
            logger.warning("⚠️ No search results found")
            return []
        
        # Group chunks by document
        document_groups = group_chunks_by_document(search_results, max_chunks_per_group)
        
        logger.info(f"✅ Document-grouped search complete: {len(document_groups)} groups ready for processing")
        return document_groups
        
    except Exception as e:
        logger.error(f"❌ Error in document-grouped search: {str(e)}")
        return []

def process_document_group(query, document_group, llm_model, system_prompt, group_index, total_groups):
    """
    Process a single document group with the LLM
    Returns the response and metadata
    """
    try:
        doc_name = document_group['document_name']
        chunks = document_group['chunks']
        
        logger.info(f"🤖 Processing group {group_index}/{total_groups}: {doc_name} ({len(chunks)} chunks)")
        
        # Build context from chunks in this group
        context = "\n\n".join([
            f"Document: {chunk['title']}\nContent: {chunk['content']}"
            for chunk in chunks
        ])
        
        # Create prompt for this document group
        group_prompt = f"""{system_prompt}

Context from relevant documents:
{context}

Current Question: {query}

EXTRACTION TASK:
Extract verbatim excerpts from the documents above that answer the current question. 
- Copy exact text from the documents
- Do not analyze, interpret, or summarize
- Provide whole sentences and complete thoughts
- Include all relevant information with proper citations
- Organize by document source

Extracted excerpts:"""
        
        # Set up LLM based on model type
        if llm_model in ["llama3-8B", "llama3-70B", "llama3.1-70B", "llama-4-Scout"]:
            if llm_model == "llama3-8B":
                api_base = os.environ.get("URL_AZURE_LLAMA3_8B")
                api_key = os.environ.get("KEY_AZURE_LLAMA3_8B")
            elif llm_model == "llama3-70B":
                api_base = os.environ.get("URL_AZURE_LLAMA3_70B")
                api_key = os.environ.get("KEY_AZURE_LLAMA3_70B")
            elif llm_model == "llama3.1-70B":
                api_base = os.environ.get("URL_AZURE_LLAMA3_1_70B")
                api_key = os.environ.get("KEY_AZURE_LLAMA3_1_70B")
            elif llm_model == "llama-4-Scout":
                api_base = os.environ.get("URL_AZURE_LLAMA4_SCOUT")
                api_key = os.environ.get("KEY_AZURE_LLAMA4_SCOUT")
            
            llm = OpenAI(
                api_base=api_base,
                api_key=api_key,
                max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4000")),
                temperature=0.1
            )
        
        elif llm_model in ["gpt-4o-mini", "gpt-4", "gpt-4o", "gpt-3.5-turbo", "o4-mini"]:
            deployment_name = get_azure_openai_deployment_name(llm_model)
            llm = AzureOpenAI(
                engine=deployment_name,
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                temperature=0.1 if not is_o_model(llm_model) else 1
            )
        
        # Save debug prompt for document group processing
        save_debug_prompt(
            group_prompt, 
            "document_group", 
            llm_model, 
            query,
            f"Group {group_index}/{total_groups}: {doc_name}, Chunks: {len(chunks)}"
        )
        
        # Generate response for this group
        response = llm.complete(group_prompt)
        
        return {
            'document_name': doc_name,
            'response': response.text,
            'chunk_count': len(chunks),
            'group_index': group_index,
            'total_groups': total_groups,
            'chunks_processed': chunks
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing document group {group_index}: {str(e)}")
        return {
            'document_name': document_group.get('document_name', 'Unknown'),
            'response': f"Error processing this document group: {str(e)}",
            'chunk_count': 0,
            'group_index': group_index,
            'total_groups': total_groups,
            'chunks_processed': [],
            'error': str(e)
        }

def aggregate_document_group_results(group_results, original_query, llm_model, system_prompt, chat_history=None):
    """
    Aggregate results from multiple document group queries
    Compile all excerpts, group by document source, preserve exact quotes and page references
    Uses the same system prompt to ensure consistency with extraction guidelines
    """
    try:
        if not group_results:
            return "No results to aggregate."
        
        logger.info(f"🔄 Aggregating results from {len(group_results)} document groups")
        
        # Build compilation prompt with all group results
        compilation_context = ""
        for group_result in group_results:
            doc_name = group_result['document_name']
            response = group_result['response']
            chunk_count = group_result['chunk_count']
            
            compilation_context += f"\n\n{'='*60}\n"
            compilation_context += f"RESULTS FROM: {doc_name} ({chunk_count} chunks processed)\n"
            compilation_context += f"{'='*60}\n"
            compilation_context += f"{response}\n"
        
        # Build chat history context for aggregation (last 4 messages to avoid token limits)
        chat_context = ""
        if chat_history:
            recent_history = chat_history[-4:]
            chat_context = "\n\nRecent Conversation Context:\n"
            for msg in recent_history:
                role = "You previously asked" if msg["role"] == "user" else "I previously answered"
                chat_context += f"{role}: {msg['content'][:200]}...\n"
        
        # Create aggregation prompt with system prompt included
        aggregation_prompt = f"""{system_prompt}

COMPILATION CONTEXT:
You are now compiling results from multiple document groups that have already been processed using the above system prompt. Each group has extracted comprehensive information from their respective document chunks.

Original Question: {original_query}

CRITICAL COMPILATION INSTRUCTIONS:
1. ORGANIZE BY INDIVIDUAL DOCUMENT: Group all excerpts by their source document (e.g., "FGD-AGAB-PHI-8.docx", "FGD-AGAB-PHI-4.docx")
2. PRESERVE VERBATIM EXCERPTS: Copy exact text from the group results below - DO NOT rewrite, analyze, or summarize
3. NO ANALYSIS: Do not provide analysis, interpretation, or synthesis - just present the raw excerpts
4. MAINTAIN CITATIONS: Keep all [file name - year] references exactly as they appear
5. COMPREHENSIVE COVERAGE: Include ALL relevant excerpts from ALL document groups

FORMAT REQUIREMENTS:
- Use clear document headings: "=== DOCUMENT: [filename] ==="
- Present excerpts as bullet points or numbered lists
- Keep exact quotes with their original citations
- Do not add commentary or analysis

Results from Document Groups:
{compilation_context}

Previous chat context:
{chat_context}

Compiled Results (organized by individual document, verbatim excerpts only):"""
        
        # Set up LLM for aggregation
        if llm_model in ["llama3-8B", "llama3-70B", "llama3.1-70B", "llama-4-Scout"]:
            if llm_model == "llama3-8B":
                api_base = os.environ.get("URL_AZURE_LLAMA3_8B")
                api_key = os.environ.get("KEY_AZURE_LLAMA3_8B")
            elif llm_model == "llama3-70B":
                api_base = os.environ.get("URL_AZURE_LLAMA3_70B")
                api_key = os.environ.get("KEY_AZURE_LLAMA3_70B")
            elif llm_model == "llama3.1-70B":
                api_base = os.environ.get("URL_AZURE_LLAMA3_1_70B")
                api_key = os.environ.get("KEY_AZURE_LLAMA3_1_70B")
            elif llm_model == "llama-4-Scout":
                api_base = os.environ.get("URL_AZURE_LLAMA4_SCOUT")
                api_key = os.environ.get("KEY_AZURE_LLAMA4_SCOUT")
            
            llm = OpenAI(
                api_base=api_base,
                api_key=api_key,
                max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4000")),
                temperature=0.1
            )
        
        elif llm_model in ["gpt-4o-mini", "gpt-4", "gpt-4o", "gpt-3.5-turbo", "o4-mini"]:
            deployment_name = get_azure_openai_deployment_name(llm_model)
            llm = AzureOpenAI(
                engine=deployment_name,
                azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                temperature=0.1 if not is_o_model(llm_model) else 1
            )
        
        # Save debug prompt for aggregation
        save_debug_prompt(
            aggregation_prompt, 
            "aggregation", 
            llm_model, 
            original_query,
            f"Group results: {len(group_results)} groups, Chat history: {len(chat_history) if chat_history else 0} messages"
        )
        
        # Generate aggregated response
        aggregated_response = llm.complete(aggregation_prompt)
        
        logger.info(f"✅ Aggregation complete: {len(group_results)} groups compiled")
        return aggregated_response.text
        
    except Exception as e:
        logger.error(f"❌ Error aggregating document group results: {str(e)}")
        return f"Error during aggregation: {str(e)}"

def process_document_groups_sequentially(query, document_groups, llm_model, system_prompt, progress_callback=None, chat_history=None):
    """
    Process each document group sequentially with progress tracking
    Returns aggregated results from all groups
    """
    try:
        if not document_groups:
            return "No document groups to process."
        
        total_groups = len(document_groups)
        group_results = []
        
        logger.info(f"🚀 Starting sequential processing of {total_groups} document groups")
        
        for i, document_group in enumerate(document_groups, 1):
            # Update progress if callback provided
            if progress_callback:
                progress_callback(i, total_groups, document_group['document_name'])
            
            # Process this document group
            group_result = process_document_group(
                query, 
                document_group, 
                llm_model, 
                system_prompt, 
                i, 
                total_groups
            )
            
            group_results.append(group_result)
            
            # Log progress
            logger.info(f"✅ Completed group {i}/{total_groups}: {document_group['document_name']}")
        
        # Aggregate all results
        logger.info("🔄 Starting aggregation of all group results")
        final_response = aggregate_document_group_results(group_results, query, llm_model, system_prompt, chat_history)
        
        logger.info(f"🎉 Multi-prompt processing complete: {total_groups} groups processed and aggregated")
        return final_response, group_results
        
    except Exception as e:
        logger.error(f"❌ Error in sequential document group processing: {str(e)}")
        return f"Error during multi-prompt processing: {str(e)}", []

def build_hybrid_context(chat_history, max_initial_user_inputs=3, max_recent_messages=6):
    """
    Build hybrid context that preserves initial user inputs and their responses, plus recent conversation
    
    Smart logic: Only includes assistant responses ≤1000 characters as "instruction confirmations".
    Longer responses indicate real questions, so those user inputs are excluded from initial context.
    
    Args:
        chat_history: List of chat messages
        max_initial_user_inputs: Number of initial user inputs to preserve (default: 3)
        max_recent_messages: Number of recent messages to include (default: 6)
    
    Returns:
        Tuple of (initial_context, recent_context, total_messages_used)
    """
    if not chat_history:
        return "", "", 0
    
    total_messages = len(chat_history)
    
    # Find the first N user inputs and their corresponding responses
    # Skip responses longer than 1000 chars (likely real questions, not instructions)
    user_inputs_found = 0
    initial_messages = []
    
    for i, msg in enumerate(chat_history):
        if msg["role"] == "user":
            user_inputs_found += 1
            if user_inputs_found > max_initial_user_inputs:
                break
            initial_messages.append(msg)
        elif msg["role"] == "assistant":
            # Only include assistant response if it's short (likely instruction confirmation)
            if len(msg["content"]) <= 1000:
                initial_messages.append(msg)
            else:
                # Long response indicates this was a real question, not an instruction
                # Remove the corresponding user input and stop collecting initial context
                if initial_messages and initial_messages[-1]["role"] == "user":
                    initial_messages.pop()  # Remove the user input
                break
        # Skip any other message types (shouldn't happen in normal chat)
    
    # Build initial context (first N user inputs + their responses)
    initial_context = ""
    if initial_messages:
        initial_context = "\n\nInitial Instructions:\n"
        for msg in initial_messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            initial_context += f"{role}: {msg['content']}\n\n"
    
    # Build recent context (remaining messages after initial instructions)
    recent_context = ""
    remaining_messages = chat_history[len(initial_messages):]
    recent_messages = []
    
    if remaining_messages:
        # Take last N recent messages from the remaining ones
        recent_messages = remaining_messages[-max_recent_messages:]
        
        recent_context = "\n\nRecent Conversation:\n"
        for msg in recent_messages:
            role = "Human" if msg["role"] == "user" else "Assistant"
            recent_context += f"{role}: {msg['content']}\n\n"
    
    total_used = len(initial_messages) + len(recent_messages)
    
    return initial_context, recent_context, total_used

def condense_question_with_context(current_query, chat_history, llm_model):
    """
    Condense the current question with hybrid context to create a standalone query
    """
    if not chat_history:
        return current_query
    
    # Build hybrid context (initial instructions + recent conversation)
    initial_context, recent_context, total_used = build_hybrid_context(chat_history, max_initial_user_inputs=3, max_recent_messages=6)
    
    # Combine contexts
    full_context = initial_context + recent_context
    
    condense_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures the full context and intent.

Conversation History:
{full_context}

Follow-up Question: {current_query}

Standalone Question:"""

    logger.info(f"🔄 Condensing query with hybrid context from {total_used} messages (initial + recent)")
    
    # Set up LLM for condensing
    if llm_model in ["llama3-8B", "llama3-70B", "llama3.1-70B", "llama-4-Scout"]:
        # Get the appropriate API endpoints based on model
        if llm_model == "llama3-8B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_8B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_8B")
        elif llm_model == "llama3-70B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_70B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_70B")
        elif llm_model == "llama3.1-70B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_1_70B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_1_70B")
        elif llm_model == "llama-4-Scout":
            api_base = os.environ.get("URL_AZURE_LLAMA4_SCOUT")
            api_key = os.environ.get("KEY_AZURE_LLAMA4_SCOUT")
        
        llm = OpenAI(
            api_base=api_base,
            api_key=api_key,
            max_tokens=200,  # Keep it short for condensing
            temperature=0.1
        )
    
    elif llm_model in ["gpt-4o-mini", "gpt-4", "gpt-4o", "gpt-3.5-turbo", "o4-mini"]:
        deployment_name = get_azure_openai_deployment_name(llm_model)
        llm = AzureOpenAI(
            engine=deployment_name,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            max_tokens=200,
            temperature=0.1 if not is_o_model(llm_model) else 1
        )
    
    try:
        # Save debug prompt for condensing
        save_debug_prompt(
            condense_prompt, 
            "condensing", 
            llm_model, 
            current_query,
            f"Chat history length: {len(chat_history)}, Total context messages: {total_used}"
        )
        
        condensed_response = llm.complete(condense_prompt)
        condensed_query = condensed_response.text.strip()
        
        # Fallback to original if condensing fails
        if not condensed_query or len(condensed_query) < 10:
            logger.warning("⚠️ Condensing produced short result, using original query")
            return current_query
            
        logger.info(f"✅ Condensed query: {condensed_query[:100]}...")
        return condensed_query
        
    except Exception as e:
        logger.error(f"❌ Error condensing query: {str(e)}")
        return current_query

def save_debug_prompt(prompt_content, query_type, model_name, query, additional_info=""):
    """
    Save the full prompt being sent to the LLM for debugging
    Only saves if DEBUG_VARIABLE environment variable is set to 1
    """
    # Check if debug saving is enabled
    if os.environ.get("DEBUG_VARIABLE", "0") != "1":
        logger.info("🔍 Debug prompt saving disabled (DEBUG_VARIABLE != 1)")
        return None
        
    try:
        # Create debug directory in project root if it doesn't exist
        debug_dir = Path("./debug")
        debug_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')
        filename = f"prompt_{query_type}_{model_name}_{safe_query}_{timestamp}.txt"
        filepath = debug_dir / filename
        
        # Write debug information
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("LLM PROMPT DEBUG - FULL PROMPT SENT TO API\n")
            f.write("=" * 80 + "\n")
            f.write(f"Query Type: {query_type}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Original Query: {query}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Additional Info: {additional_info}\n")
            f.write("=" * 80 + "\n\n")
            f.write("FULL PROMPT CONTENT:\n")
            f.write("-" * 40 + "\n")
            f.write(prompt_content)
            f.write("\n" + "-" * 40 + "\n")
            f.write(f"\nPrompt Length: {len(prompt_content)} characters\n")
            f.write(f"Prompt Word Count: {len(prompt_content.split())} words\n")
        
        logger.info(f"🔍 DEBUG: Saved prompt to: {filepath.absolute()}")
        return str(filepath.absolute())
        
    except Exception as e:
        logger.error(f"❌ Error saving debug prompt: {str(e)}")
        return None

def save_debug_chunks(search_results, query, container_name):
    """
    Save retrieved chunks to a debug file for debugging
    Only saves if DEBUG_VARIABLE environment variable is set to 1
    """
    # Check if debug saving is enabled
    if os.environ.get("DEBUG_VARIABLE", "0") != "1":
        logger.info("🔍 Debug saving disabled (DEBUG_VARIABLE != 1)")
        return None
        
    try:
        # Create debug directory in project root if it doesn't exist
        debug_dir = Path("./debug")
        debug_dir.mkdir(exist_ok=True)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')
        filename = f"chunks_{container_name}_{safe_query}_{timestamp}.txt"
        filepath = debug_dir / filename
        
        # Write debug information
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AZURE AI SEARCH DEBUG - RETRIEVED CHUNKS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Query: {query}\n")
            f.write(f"Container: {container_name}\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total chunks retrieved: {len(search_results)}\n")
            f.write("=" * 80 + "\n\n")
            
            # Group chunks by document
            chunks_by_doc = {}
            for chunk in search_results:
                doc_name = chunk.get('title', 'Unknown')
                if doc_name not in chunks_by_doc:
                    chunks_by_doc[doc_name] = []
                chunks_by_doc[doc_name].append(chunk)
            
            # Write chunks grouped by document
            for doc_name, chunks in chunks_by_doc.items():
                f.write(f"\n{'='*60}\n")
                f.write(f"DOCUMENT: {doc_name}\n")
                f.write(f"Chunks from this document: {len(chunks)}\n")
                f.write(f"{'='*60}\n\n")
                
                for i, chunk in enumerate(chunks, 1):
                    f.write(f"--- CHUNK {i} ---\n")
                    f.write(f"Score: {chunk.get('score', 'N/A')}\n")
                    f.write(f"Filepath: {chunk.get('filepath', 'N/A')}\n")
                    f.write(f"Content:\n{chunk.get('content', 'No content')}\n")
                    f.write(f"\n{'---'*20}\n\n")
            
            # Write summary statistics
            f.write(f"\n{'='*80}\n")
            f.write("SUMMARY STATISTICS\n")
            f.write(f"{'='*80}\n")
            f.write(f"Total documents consulted: {len(chunks_by_doc)}\n")
            f.write(f"Total chunks retrieved: {len(search_results)}\n")
            
            # Document breakdown
            f.write(f"\nDocument breakdown:\n")
            for doc_name, chunks in chunks_by_doc.items():
                f.write(f"  - {doc_name}: {len(chunks)} chunks\n")
            
            # Score statistics
            scores = [chunk.get('score', 0) for chunk in search_results if chunk.get('score') is not None]
            if scores:
                f.write(f"\nScore statistics:\n")
                f.write(f"  - Min score: {min(scores):.4f}\n")
                f.write(f"  - Max score: {max(scores):.4f}\n")
                f.write(f"  - Avg score: {sum(scores)/len(scores):.4f}\n")
        
        logger.info(f"🔍 DEBUG: Saved {len(search_results)} chunks to: {filepath.absolute()}")
        return str(filepath.absolute())
        
    except Exception as e:
        logger.error(f"❌ Error saving debug chunks: {str(e)}")
        return None

def generate_llm_response_with_context(query, search_results, llm_model, system_prompt, chat_history):
    """Enhanced response generation that includes hybrid chat context"""
    
    # Build context from search results
    context = "\n\n".join([
        f"Document: {doc['title']}\nContent: {doc['content']}"
        for doc in search_results
    ])
    
    # Build hybrid chat context (initial instructions + recent conversation)
    chat_context = ""
    if chat_history:
        initial_context, recent_context, total_used = build_hybrid_context(chat_history, max_initial_user_inputs=3, max_recent_messages=4)
        chat_context = initial_context + recent_context
        
        # Truncate recent context if needed (but preserve initial context)
        if len(chat_context) > 2000:  # Rough token limit
            # Keep initial context intact, truncate recent context
            initial_lines = initial_context.split('\n')
            recent_lines = recent_context.split('\n')
            
            # Rebuild with truncated recent context
            chat_context = '\n'.join(initial_lines)
            truncated_recent = '\n'.join(recent_lines[:len(recent_lines)//2])  # Truncate recent context
            chat_context += truncated_recent
    
    # Create enhanced prompt with both document context and hybrid chat history
    full_prompt = f"""{system_prompt}

Context from relevant documents:
{context}
{chat_context}

Current Question: {query}

Answer the current question based on the provided document context, and consider the conversation context for continuity:"""
    
    logger.info(f"🤖 Generating response with {llm_model} (including hybrid chat context)")
    
    # Set up LLM based on model type (same as before)
    if llm_model in ["llama3-8B", "llama3-70B", "llama3.1-70B", "llama-4-Scout"]:
        if llm_model == "llama3-8B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_8B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_8B")
        elif llm_model == "llama3-70B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_70B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_70B")
        elif llm_model == "llama3.1-70B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_1_70B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_1_70B")
        elif llm_model == "llama-4-Scout":
            api_base = os.environ.get("URL_AZURE_LLAMA4_SCOUT")
            api_key = os.environ.get("KEY_AZURE_LLAMA4_SCOUT")
        
        llm = OpenAI(
            api_base=api_base,
            api_key=api_key,
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4000")),
            temperature=0.1
        )
    
    elif llm_model in ["gpt-4o-mini", "gpt-4", "gpt-4o", "gpt-3.5-turbo", "o4-mini"]:
        deployment_name = get_azure_openai_deployment_name(llm_model)
        llm = AzureOpenAI(
            engine=deployment_name,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.1 if not is_o_model(llm_model) else 1
        )
    
    # Save debug prompt for main response generation
    save_debug_prompt(
        full_prompt, 
        "main_response", 
        llm_model, 
        query,
        f"Search results: {len(search_results)} chunks, Chat history: {len(chat_history) if chat_history else 0} messages"
    )
    
    # Generate response
    response = llm.complete(full_prompt)
    return response.text

def generate_chat_only_response(query, llm_model, chat_history):
    """Generate response without document search - chat only mode with hybrid context"""
    
    # Build hybrid chat context (initial instructions + recent conversation)
    chat_context = ""
    if chat_history:
        initial_context, recent_context, total_used = build_hybrid_context(chat_history, max_initial_user_inputs=3, max_recent_messages=6)
        chat_context = initial_context + recent_context
    
    # Create chat-only prompt
    chat_prompt = f"""You are an expert research assistant. You are assisting a formative evaluation of UNICEF East Asia Pacific's adolescent girls–focused programming for the period of 2022–2025. Your task is to review selected documents in this knowledge base and extract verbatim excerpts or exact text to answer specific questions. You are currently in chat-only mode, which means you are not searching any documents. Please respond to the user's question based on your general knowledge and the conversation context.

When answering a question:
1. COMPREHENSIVENESS: Always search across all available information
2. EXCLUSION: Only exclude background or information that does not connect to the question at all. Do not: interpret, analyse, infer, or add any external information.
3. COMPLETENESS: Include any relevant information to answer the question. Do not skip any information.

{chat_context}

Current Question: {query}

Please provide a helpful response. If the user is asking about documents or research, kindly remind them that they are in chat-only mode and suggest they enable document search if they need information from the knowledge base."""
    
    logger.info(f"🤖 Generating chat-only response with {llm_model} (hybrid context from {len(chat_history)} messages)")
    
    # Set up LLM based on model type
    if llm_model in ["llama3-8B", "llama3-70B", "llama3.1-70B", "llama-4-Scout"]:
        if llm_model == "llama3-8B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_8B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_8B")
        elif llm_model == "llama3-70B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_70B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_70B")
        elif llm_model == "llama3.1-70B":
            api_base = os.environ.get("URL_AZURE_LLAMA3_1_70B")
            api_key = os.environ.get("KEY_AZURE_LLAMA3_1_70B")
        elif llm_model == "llama-4-Scout":
            api_base = os.environ.get("URL_AZURE_LLAMA4_SCOUT")
            api_key = os.environ.get("KEY_AZURE_LLAMA4_SCOUT")
        
        llm = OpenAI(
            api_base=api_base,
            api_key=api_key,
            max_tokens=int(os.environ.get("OPENAI_MAX_TOKENS", "4000")),
            temperature=0.1
        )
    
    elif llm_model in ["gpt-4o-mini", "gpt-4", "gpt-4o", "gpt-3.5-turbo", "o4-mini"]:
        deployment_name = get_azure_openai_deployment_name(llm_model)
        llm = AzureOpenAI(
            engine=deployment_name,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            temperature=0.1 if not is_o_model(llm_model) else 1
        )
    
    # Save debug prompt for chat-only response
    save_debug_prompt(
        chat_prompt, 
        "chat_only", 
        llm_model, 
        query,
        f"Chat history: {len(chat_history) if chat_history else 0} messages"
    )
    
    # Generate response
    response = llm.complete(chat_prompt)
    return response.text

# IMPORTANT: Initialize session state for user isolation
if "messages" not in st.session_state:
    st.session_state.messages = []
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
if "similarity_top_k" not in st.session_state:
    st.session_state.similarity_top_k = 50
if "use_query_optimization" not in st.session_state:
    st.session_state.use_query_optimization = False
if "selected_documents" not in st.session_state:
    st.session_state.selected_documents = set()
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """System prompt:
You are an expert research assistant. You are assisting a formative evaluation of UNICEF East Asia Pacific's adolescent girls–focused programming for the period of 2022–2025. Your task is to review selected documents in this knowledge base and extract verbatim excerpts or exact text to answer specific questions.
When answering a question:
1. COMPREHENSIVENESS: Always search across ALL selected documents in the knowledge base and extract information from each of them individually. You must extract responses from each selected document for each question. 
2. STRUCTURED RESPONSES: Copy exact information from the selected documents to answer the question. Organize answers in excerpts or paragraphs from the text. Provide whole sentences as a standard rule. Where there are no complete sentences, include all relevant information to answer the question in a paragraph. 
3. CITATIONS: Add a simple reference to each extraction using the format [file name - year (if applicable)]. DO NOT INCLUDE THE PAGE. 
4. DETAIL LEVEL: Extract relevant information that answers the question in detail. Provide evidence (such as data/results, programs/partners, implementation details, challenges, dates/locations), or context that is tied to the question where it is present. 
5. EXCLUSION: Only exclude background or information that does not connect to the question at all. Do not: interpret, analyse, infer, or add any external information.
6. COMPLETENESS: Include any relevant information from the documents to answer the question. Do not skip any information.

Points to remember: 
1. Multi-country documents: Some documents in this knowledge base may be global or mention multitple countries. Always extract information only for countries in the EAST-ASIA PACIFIC region. Within this region, explicit instructions will be given in the chat when you need to extract information for one specific country or group of countries. 
2. Acronyms: The documents in this knowledge base contain some commonly used abbreviations and some specific acronyms related to this assignment. You MUST extract text to answer evaluation questions along with acronyms, where applicable. Do NOT leave out any excerpts if you do not understand the acronyms, simply extract as it is. 
Commonly used acronym types and list to look out for (including but not limited to): 
Population sub-groups: AG = adolescent girls, AB = adolescent boys, and others such as HW, YP…
UNICEF areas of work: CP = child protection, ALS = alternate learning systems, and others such as OOS, CSE, SP, VAC, VAW, GBV, CEFMU, CM, SBC, MHM, NFE, ECCD, GE, GR…
Offices, Departments and Other Occupational Units: CO = Country Office, FP = focal point, DoE = Department of Education and others such as RO, DoH, MoH, NAP, RHU, LGU, GAWG, IP…
   """
if "search_documents_enabled" not in st.session_state:
    st.session_state.search_documents_enabled = False

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

    if os.environ.get("DETAILED_LOGGING", 1) == 1:
    
        logging.getLogger("llama_index").setLevel(logging.DEBUG)
        logging.getLogger("openai").setLevel(logging.DEBUG)

  
    blob_service_client = BlobServiceClient.from_connection_string(f"DefaultEndpointsProtocol=https;AccountName={azure_storage_account_name};AccountKey={azure_storage_account_key}")

    container_list = list_all_containers()
    container_list = [container for container in container_list if container.startswith("genai")]

    st.sidebar.error("1️⃣ Select your knowledge base and model")
    container_name = st.sidebar.selectbox("Knowledge base choice: ", container_list)
    model_variable = st.sidebar.selectbox("Model choice: ", ["gpt-4o","o4-mini","gpt-4","llama-4-Scout"])
    deep_research = st.sidebar.checkbox(
    "🔎 Deep Research",
    value=False,
    key="deep_research",
    help="Enable it when going through many or heavy documents"
)

    # Set default values without UI controls
    use_multi_prompt = True  # Multi-prompt strategy ON by default
    mmr_enabled = False      # MMR diversity OFF by default
    





    #QUERY OPTIMIZATION - TOP K
    # with st.sidebar:
    #     st.markdown("---")
    #     st.subheader("🎯 Query Optimization")
    #     st.session_state.use_query_optimization = st.toggle(
    #         "Smart Query Optimization", 
    #         value=st.session_state.use_query_optimization,
    #         help="Automatically adjusts the number of document chunks retrieved based on query complexity"
    #     )
        
    #     if st.session_state.use_query_optimization:
    #         st.info("🧠 Each query will be analyzed for optimal retrieval")
    #     else:
    #         st.info(f"Using fixed retrieval parameters ")


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

    # Document selection interface
    st.sidebar.markdown("---")
    st.sidebar.subheader("📄 Select Documents to Search - No more than 10")

    blob_list = list_all_files(container_name)

    # Initialize selected documents if container changed or first time
    if (st.session_state.current_container != container_name or 
        not hasattr(st.session_state, 'selected_documents')):
        # Select all documents by default
        st.session_state.selected_documents = {doc["Name"] for doc in blob_list}

    # Initialize action flags for button clicks
    if "select_all_clicked" not in st.session_state:
        st.session_state.select_all_clicked = False
    if "deselect_all_clicked" not in st.session_state:
        st.session_state.deselect_all_clicked = False

    # Add Select All / Deselect All buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("✅ Select All", help="Select all available documents", key="select_all_btn"):
            st.session_state.select_all_clicked = True
            st.session_state.deselect_all_clicked = False

    with col2:
        if st.button("❌ Deselect All", help="Deselect all documents", key="deselect_all_btn"):
            st.session_state.deselect_all_clicked = True
            st.session_state.select_all_clicked = False

    # Apply button actions
    if st.session_state.select_all_clicked:
        st.session_state.selected_documents = {doc["Name"] for doc in blob_list}
        st.session_state.select_all_clicked = False  # Reset flag

    if st.session_state.deselect_all_clicked:
        st.session_state.selected_documents = set()
        st.session_state.deselect_all_clicked = False  # Reset flag

    # Document selection checkboxes
    selected_docs_temp = set()
    for doc in blob_list:
        doc_name = doc["Name"]
        
        # Checkbox reflects current selection state
        is_selected = st.sidebar.checkbox(
            doc_name,
            value=doc_name in st.session_state.selected_documents,
            key=f"doc_checkbox_{doc_name}"
        )
        
        if is_selected:
            selected_docs_temp.add(doc_name)

    # Update session state with current checkbox states
    st.session_state.selected_documents = selected_docs_temp

    # Show selection summary
    if st.session_state.selected_documents:
        chunks_per_doc = 100 // len(st.session_state.selected_documents) if deep_research else 70 // len(st.session_state.selected_documents)
        st.sidebar.success(f"✅ {len(st.session_state.selected_documents)} document(s) selected")
        st.sidebar.info(f"📊 ~{chunks_per_doc} chunks per document")
        st.sidebar.info(f"📊 Minimum: 5 chunks per document")
    else:
        st.sidebar.warning("⚠️ No documents selected")
  

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

    # allow for parameter update 
    st.error(" 2️⃣ Adjust model's parameters if needed (for advanced use only) - Do not update them once you started chatting")
    with st.expander("🔧 Advanced Parameters", expanded=False):              

        # System Prompt configuration
        col1, col2 = st.columns([0.95, 0.05])
        with col1:
            system_prompt = st.text_area(
                "System Prompt - Define how the AI should behave and respond to queries",
                value=st.session_state.system_prompt,
                height=300,
                key="system_prompt_input"
                
            )
        with col2:
            st.markdown(" ", help="""**What is a System Prompt?**

        The system prompt defines the AI's behavior, personality, and response style. It acts as instructions that guide how the model interprets and answers your questions.

        **How to define a good system prompt:**
        • Be specific about the desired output format
        • Define the level of detail you want
        • Specify citation requirements
        • Include any domain-specific expertise needed
        • Set the tone (formal, casual, technical, etc.)
        • Mention any constraints or requirements

        **Examples:**
        • "Be concise and technical" for brief answers
        • "Provide detailed explanations with examples" for comprehensive responses
        • "Focus on financial implications" for business documents""")

        # Set data_loaded to True when we have container and model
        if container_name and model_variable:
            st.session_state.data_loaded = True
            st.session_state.current_container = container_name
            st.session_state.current_model = model_variable
            st.success("✅ Ready to search existing Azure index!")



    # DIRECT AZURE SEARCH CHAT INTERFACE
    if st.session_state.data_loaded:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])


        # # Search mode toggle - positioned BEFORE chat input
        # col1, col2 = st.columns([0.5, 0.5])
        
        # with col1:
        #     search_mode = st.toggle(
        #         "🔍 Search Documents",
        #         value=st.session_state.search_documents_enabled,
        #         help="Toggle to search documents or chat without searching"
        #     )
        #     st.session_state.search_documents_enabled = search_mode

        # Chat input
        # Chat input (replace your existing chat input section with this)
        if prompt := st.chat_input("Your question"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            logger.info(f"User asked: {prompt}")
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)


        if prompt:
            # Generate assistant response
            with st.chat_message("assistant"):
                try:
                    # Check if document search is enabled
                    if st.session_state.search_documents_enabled:
                        # Step 1: Condense query with chat history context
                        with st.spinner("🔄 Analyzing conversation context..."):
                            condensed_query = condense_question_with_context(
                                prompt, 
                                st.session_state.messages[:-1],  # Exclude current message
                                model_variable
                            )
                            
                            # Show condensed query if different from original
                            if condensed_query != prompt:
                                with st.expander("🔍 Condensed Query", expanded=False):
                                    st.write(f"**Original:** {prompt}")
                                    st.write(f"**Condensed:** {condensed_query}")
                        
                        # Step 2: Choose Processing Strategy
                        if use_multi_prompt:
                            # Multi-Prompt Document-Grouped Search and Processing
                            with st.spinner("🔍 Searching and grouping documents..."):
                                # Calculate total chunks based on deep research setting
                                total_chunks = 100 if deep_research else 70
                                max_chunks_per_group = 20  # Document-grouped strategy limit
                                
                                # Debug: Show what we're searching
                                st.write(f"🔍 **Searching** {len(st.session_state.selected_documents)} documents: {list(st.session_state.selected_documents)}")
                                st.write(f"📊 **Total chunks to retrieve:** {total_chunks}")
                                # st.write(f"📊 **Max chunks per group:** {max_chunks_per_group}")
                                # st.info("🔄 Using Multi-Prompt Strategy: Document-grouped processing to avoid context overload")
                                
                                # Use document-grouped search
                                document_groups = search_document_groups(
                                    container_name, 
                                    condensed_query,  # Use condensed query for search
                                    st.session_state.selected_documents,
                                    total_chunks,
                                    min_chunks_per_doc=5,  # Ensure minimum 5 chunks per document
                                    max_chunks_per_group=max_chunks_per_group,
                                    mmr_enabled=mmr_enabled  # Use UI toggle setting
                                )
                                
                                if not document_groups:
                                    response_text = "No relevant documents found for your query."
                                    st.write(response_text)
                                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                                else:
                                    # Show document groups info
                                    total_chunks_retrieved = sum(group['chunk_count'] for group in document_groups)
                                    all_docs_included = set()
                                    for group in document_groups:
                                        docs_included = group.get('documents_included', [group['document_name']])
                                        all_docs_included.update(docs_included)
                                    st.info(f"🔍 Searched {len(st.session_state.selected_documents)} selected documents • Retrieved {total_chunks_retrieved} chunks from {len(all_docs_included)} documents • Created {len(document_groups)} processing groups")
                                    
                                    # Save debug information for all chunks
                                    all_chunks = []
                                    for group in document_groups:
                                        all_chunks.extend(group['chunks'])
                                    debug_file = save_debug_chunks(all_chunks, condensed_query, container_name)
                                    if debug_file:
                                        st.success(f"🔍 Debug: Chunks saved to {debug_file}")
                                        logger.info(f"🔍 DEBUG FILE: {debug_file}")
                                    
                                    # Step 3: Multi-Prompt Processing with Progress Tracking
                                    progress_container = st.container()
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    def update_progress(current_group, total_groups, doc_name):
                                        progress = current_group / total_groups
                                        progress_bar.progress(progress)
                                        status_text.info(f"🤖 Processing group {current_group}/{total_groups}: {doc_name}")
                                    
                                    with st.spinner("🤖 Processing document groups with multi-prompt strategy..."):
                                        response_text, group_results = process_document_groups_sequentially(
                                            prompt,  # Use original query for response generation
                                            document_groups,
                                            model_variable, 
                                            st.session_state.system_prompt,
                                            progress_callback=update_progress,
                                            chat_history=st.session_state.messages[:-1]  # Pass chat history
                                        )
                                    
                                    # Clear progress indicators
                                    progress_bar.empty()
                                    status_text.empty()
                                    
                                    # Show processing summary
                                    successful_groups = len([r for r in group_results if 'error' not in r])
                                    total_chunks_processed = sum(r.get('chunk_count', 0) for r in group_results)
                                    st.success(f"✅ Multi-prompt processing complete: {successful_groups}/{len(document_groups)} groups processed successfully ({total_chunks_processed} total chunks)")
                                    
                                    # Show detailed group results in expandable section
                                    with st.expander("📊 Detailed Processing Results", expanded=False):
                                        for result in group_results:
                                            status = "✅" if 'error' not in result else "❌"
                                            docs_included = result.get('documents_included', [result['document_name']])
                                            docs_str = ', '.join(docs_included) if len(docs_included) > 1 else docs_included[0]
                                            st.write(f"{status} **{result['document_name']}**: {result.get('chunk_count', 0)} chunks from {len(docs_included)} document(s)")
                                            if len(docs_included) > 1:
                                                st.caption(f"   Documents: {docs_str}")
                                            if 'error' in result:
                                                st.error(f"Error: {result['error']}")
                                    
                                    st.write(response_text)
                                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                        else:
                            # Original Single-Prompt Strategy
                            with st.spinner("🔍 Searching selected documents..."):
                                # Calculate total chunks based on deep research setting
                                total_chunks = 100 if deep_research else 70
                                
                                # Debug: Show what we're searching
                                st.write(f"🔍 **Debug:** Searching {len(st.session_state.selected_documents)} documents: {list(st.session_state.selected_documents)}")
                                st.write(f"📊 **Total chunks to retrieve:** {total_chunks}")
                                st.info("📝 Using Single-Prompt Strategy: Traditional RAG processing")
                                
                                # Use multi-document search with selected documents
                                search_results = search_azure_with_document_filter(
                                    container_name, 
                                    condensed_query,  # Use condensed query for search
                                    st.session_state.selected_documents,
                                    total_chunks,
                                    min_chunks_per_doc=5,  # Ensure minimum 5 chunks per document
                                    mmr_enabled=mmr_enabled,  # Use UI toggle setting
                                    mmr_lambda=0.5  # Fixed balanced setting
                                )
                                
                                if not search_results:
                                    response_text = "No relevant documents found for your query."
                                    st.write(response_text)
                                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                                else:
                                    # Show search results info
                                    docs_consulted = len(set(result['title'] for result in search_results))
                                    st.info(f"🔍 Searched {len(st.session_state.selected_documents)} selected documents • Retrieved {len(search_results)} chunks from {docs_consulted} documents")
                                    
                                    # Save debug information
                                    debug_file = save_debug_chunks(search_results, condensed_query, container_name)
                                    if debug_file:
                                        st.success(f"🔍 Debug: Chunks saved to {debug_file}")
                                        logger.info(f"🔍 DEBUG FILE: {debug_file}")
                                    
                                    # Step 3: Generate response with both document context and chat history
                                    with st.spinner("🤖 Generating response..."):
                                        response_text = generate_llm_response_with_context(
                                            prompt,  # Use original query for response generation
                                            search_results, 
                                            model_variable, 
                                            st.session_state.system_prompt,
                                            st.session_state.messages[:-1]  # Pass chat history
                                        )
                                    
                                    st.write(response_text)
                                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    else:
                        # Chat-only mode - no document search
                        with st.spinner("💬 Generating chat response..."):
                            st.info("💬 **Chat Only Mode** - No documents searched")
                            response_text = generate_chat_only_response(
                                prompt,
                                model_variable,
                                st.session_state.messages[:-1]  # Pass chat history
                            )
                            st.write(response_text)
                            st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Chat error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


    # Clear chat history button
    # if st.session_state.messages:
    #     if st.button("🗑️ Clear Chat History"):
    #         st.session_state.messages = []
    #         st.rerun()
    col1, col2 = st.columns([0.75, 0.25])
    
    with col1:
        # Left-align content in col1
        st.markdown('<div style="text-align: left;">', unsafe_allow_html=True)
        search_mode = st.toggle(
            "🔍 Search Documents",
            value=st.session_state.search_documents_enabled,
            help="Toggle to search documents or chat without searching"
        )
        st.session_state.search_documents_enabled = search_mode
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        # Right-align content in col2
        st.markdown('<div style="display: flex; justify-content: flex-end;">', unsafe_allow_html=True)
        # Clear chat history button
        if st.session_state.messages:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.error("Please enter the correct password to access the application.")
