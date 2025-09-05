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
        
        logger.info(f"🔍 Searching Azure index: {index_name}")
        
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
        
        # Search with hybrid approach (vector + keyword)
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            hybrid_search=HybridSearch(),
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

def search_azure_with_document_filter(container_name, query, selected_documents, total_chunks=40, min_chunks_per_doc=5, mmr_enabled=True, mmr_lambda=0.5):
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
        
        logger.info(f"🔍 Optimized multi-document search: {len(selected_documents)} documents, {total_chunks} total chunks, min {min_chunks_per_doc} per doc")
        
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
        
        # Single search call for all selected documents
        results = search_client.search(
            search_text=query,
            vector_queries=[vector_query],
            hybrid_search=HybridSearch(),
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

def condense_question_with_context(current_query, chat_history, llm_model):
    """
    Condense the current question with chat history context to create a standalone query
    """
    if not chat_history:
        return current_query
    
    # Take last 6 messages (3 exchanges) to avoid token limits
    recent_history = chat_history[-6:]
    
    # Format chat history
    history_text = ""
    for msg in recent_history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n\n"
    
    condense_prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that captures the full context and intent.

Conversation History:
{history_text}

Follow-up Question: {current_query}

Standalone Question:"""

    logger.info(f"🔄 Condensing query with context from {len(recent_history)} previous messages")
    
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

def generate_llm_response_with_context(query, search_results, llm_model, system_prompt, chat_history):
    """Enhanced response generation that includes chat context"""
    
    # Build context from search results
    context = "\n\n".join([
        f"Document: {doc['title']}\nContent: {doc['content']}"
        for doc in search_results
    ])
    
    # Build chat history context (last 4 messages to avoid token limits)
    chat_context = ""
    if chat_history:
        recent_history = chat_history[-4:]
        chat_context = "\n\nRecent Conversation:\n"
        for msg in recent_history:
            role = "You previously asked" if msg["role"] == "user" else "I previously answered"
            chat_context += f"{role}: {msg['content'][:200]}...\n"
    
    # Create enhanced prompt with both document context and chat history
    full_prompt = f"""{system_prompt}

Context from relevant documents:
{context}
{chat_context}

Current Question: {query}

Answer the current question based on the provided document context, and consider the recent conversation context for continuity:"""
    
    logger.info(f"🤖 Generating response with {llm_model} (including chat context)")
    
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
    
    # Generate response
    response = llm.complete(full_prompt)
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
You are an expert research assistant. You are assisting a literature review for a formative evaluation of UNICEF East Asia Pacific’s adolescent girls–focused programming for the period of 2022–2025. Your task is to review selected documents in this knowledge base and extract verbatim excerpts or exact text to answer specific questions.
When answering a question:
1. COMPREHENSIVENESS: Always search across ALL selected documents in the knowledge base and extract information from each of them individually. You must extract responses from each selected document for each question. 
2. STRUCTURED RESPONSES: Copy exact information from the selected documents to answer the question. Organize answers in short excerpts or bullet points of the text.
3. CITATIONS: Add a simple reference to each extraction using the format [file name - year (if applicable)]. If the same point appears in multiple selected documents, list it once and append all references at the end. DO NOT INCLUDE THE PAGE
4. DETAIL LEVEL: Only extract what answers the question. Provide evidence (such as data/results, programs/partners, implementation details, dates/locations), or context that is tied to the question where it is present. 
5. EXCLUSION: Exclude general background or information that does not connect to the question at all. Do not: interpret, analyse, infer, or add any external information.
   """

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
    model_variable = st.sidebar.selectbox("Model choice: ", ["o4-mini","gpt-4","gpt-4o","llama-4-Scout"])
    deep_research = st.sidebar.checkbox(
    "🔎 Deep Research",
    value=False,
    key="deep_research",
    help="Enable it when going through many or heavy documents"
)





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
        not st.session_state.selected_documents):
        # Select all documents by default
        st.session_state.selected_documents = {doc["Name"] for doc in blob_list}
    
    # Document selection checkboxes
    selected_docs = []
    for doc in blob_list:
        doc_name = doc["Name"]
        is_selected = st.sidebar.checkbox(
            doc_name,
            value=doc_name in st.session_state.selected_documents,
            key=f"doc_{doc_name}"
        )
        if is_selected:
            selected_docs.append(doc_name)
    
    # Update session state with current selection
    st.session_state.selected_documents = set(selected_docs)
    
    # Show selection summary
    if selected_docs:
        chunks_per_doc = 80 // len(selected_docs) if deep_research else 50 // len(selected_docs)
        st.sidebar.success(f"✅ {len(selected_docs)} document(s) selected")
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

        # Chat input
        # Chat input (replace your existing chat input section with this)
        if prompt := st.chat_input("Your question"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            logger.info(f"User asked: {prompt}")
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)

            # Generate assistant response
            with st.chat_message("assistant"):
                try:
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
                    
                    # Step 2: Search Azure with multi-document strategy
                    with st.spinner("🔍 Searching selected documents..."):
                        # Calculate total chunks based on deep research setting
                        total_chunks = 70 if deep_research else 40
                        
                        # Debug: Show what we're searching
                        st.write(f"🔍 **Debug:** Searching {len(st.session_state.selected_documents)} documents: {list(st.session_state.selected_documents)}")
                        st.write(f"📊 **Total chunks to retrieve:** {total_chunks}")
                        
                        # Use multi-document search with selected documents
                        search_results = search_azure_with_document_filter(
                            container_name, 
                            condensed_query,  # Use condensed query for search
                            st.session_state.selected_documents,
                            total_chunks,
                            min_chunks_per_doc=5,  # Ensure minimum 5 chunks per document
                            mmr_enabled=True,  # Always enable MMR
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
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Chat error: {str(e)}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Clear chat history button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

else:
    st.error("Please enter the correct password to access the application.")
