"""
Vector Store Module

This module provides a simple vector store for storing and retrieving document chunks 
based on embedding similarity. It supports:
1. Adding PDF chunks with their embeddings
2. Semantic similarity search with Maximum Marginal Relevance (MMR)
3. Retrieval of top-k semantically relevant chunks
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, TypeVar
import faiss
from .pdf_ingestion import PDFChunk
from .web_ingestion import WebChunk

# Type variable for document chunks
T = TypeVar('T', PDFChunk, WebChunk)


class VectorStore:
    """
    A simple vector store using FAISS for efficient similarity search.
    Implements Maximum Marginal Relevance for diverse retrieval.
    Supports both PDFChunk and WebChunk objects.
    """
    
    def __init__(self, embedding_dim: int = 1536):
        """
        Initialize a vector store.
        
        Args:
            embedding_dim: Dimension of the embeddings (1536 for text-embedding-3-small)
        """
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # Inner product for cosine similarity with normalized vectors
        self.chunks: List[Union[PDFChunk, WebChunk]] = []
    
    def add_chunks(self, chunks: List[Union[PDFChunk, WebChunk]]) -> None:
        """
        Add chunks to the vector store.
        
        Args:
            chunks: List of PDFChunk or WebChunk objects with embeddings
        """
        # Filter out chunks without embeddings
        valid_chunks = [chunk for chunk in chunks if chunk.embedding is not None]
        
        if not valid_chunks:
            return
        
        # Extract embeddings and convert to numpy array
        embeddings = np.array([chunk.embedding for chunk in valid_chunks], dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to index
        self.index.add(embeddings)
        
        # Store chunks
        self.chunks.extend(valid_chunks)
    
    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Tuple[PDFChunk, float]]:
        """
        Perform similarity search to find the most similar chunks.
        
        Args:
            query_embedding: Embedding vector of the query
            k: Number of results to return
            
        Returns:
            List of tuples (chunk, score)
        """
        if not self.chunks:
            return []
        
        # Convert query embedding to numpy and normalize
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # Search
        scores, indices = self.index.search(query_np, min(k, len(self.chunks)))
        
        # Return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid index
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def mmr_search(self, 
                  query_embedding: List[float], 
                  k: int = 6, 
                  lambda_param: float = 0.5) -> List[PDFChunk]:
        """
        Maximum Marginal Relevance search for diverse results.
        
        Args:
            query_embedding: Embedding vector of the query
            k: Number of results to return
            lambda_param: Balance between relevance and diversity (0.0-1.0)
                          Higher values favor relevance, lower values favor diversity
                          
        Returns:
            List of PDFChunks
        """
        if len(self.chunks) == 0:
            return []
        
        if len(self.chunks) <= k:
            return self.chunks
        
        # Get initial similarity search results (more than we need)
        initial_k = min(2 * k, len(self.chunks))
        results = self.similarity_search(query_embedding, initial_k)
        
        # Extract embeddings from results
        doc_embeddings = np.array([chunk.embedding for chunk, _ in results], dtype=np.float32)
        
        # Normalize query embedding for cosine similarity
        query_np = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(query_np)
        
        # Calculate relevance scores
        relevance_scores = np.dot(doc_embeddings, query_np.T).flatten()
        
        # Initialize selected indices and remaining indices
        selected_indices = []
        remaining_indices = list(range(len(results)))
        
        # Select the first document with highest relevance
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        
        # Select remaining documents using MMR
        for _ in range(min(k - 1, len(results) - 1)):
            if not remaining_indices:
                break
                
            # Calculate diversity scores
            diversity_scores = np.zeros(len(remaining_indices))
            selected_embeddings = doc_embeddings[selected_indices]
            
            for i, idx in enumerate(remaining_indices):
                # Calculate similarity to already selected documents
                similarity = np.max(np.dot(doc_embeddings[idx], selected_embeddings.T))
                diversity_scores[i] = similarity
            
            # Calculate MMR scores
            mmr_scores = lambda_param * relevance_scores[remaining_indices] - \
                        (1 - lambda_param) * diversity_scores
            
            # Select the document with highest MMR score
            mmr_idx = np.argmax(mmr_scores)
            selected_idx = remaining_indices[mmr_idx]
            selected_indices.append(selected_idx)
            remaining_indices.remove(selected_idx)
        
        # Return selected chunks
        return [results[idx][0] for idx in selected_indices]


async def get_query_embedding(query: str, 
                             openai_client, 
                             embedding_model: str = "text-embedding-3-small",
                             llm_usage = None) -> List[float]:
    """
    Get embedding for a query string.
    
    Args:
        query: The query text to embed
        openai_client: OpenAI client
        embedding_model: Model to use for embeddings
        llm_usage: Optional tracker for token usage
        
    Returns:
        Embedding vector
    """
    try:
        response = await openai_client.embeddings.create(
            input=query,
            model=embedding_model,
        )
        
        # Track token usage if provided
        if llm_usage:
            total_tokens = response.usage.prompt_tokens
            await llm_usage.add_tokens(
                model=embedding_model,
                input_tokens=total_tokens,
                output_tokens=0,
                cached_tokens=0,
                reasoning_tokens=0,
                label=f"embed_query"
            )
        
        embedding = response.data[0].embedding
        return embedding
        
    except Exception as e:
        print(f"Error getting query embedding: {e}")
        return []
