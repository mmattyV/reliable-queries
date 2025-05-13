"""
PDF Ingestion Module

This module provides utilities to:
1. Parse PDFs with pdfplumber for accurate layout extraction
2. Chunk text with adaptive sizes (~350 tokens) and overlap (30%)
3. Embed chunks using OpenAI's text-embedding-3-small
4. Store chunks with metadata including page_index and character offsets
"""

import os
import re
import math
from typing import List, Dict, Any, Optional, Tuple
import pdfplumber
import numpy as np
from openai import AsyncOpenAI
import tiktoken
from dataclasses import dataclass
from .model_costs import ModelUsageAsync


@dataclass
class PDFChunk:
    """Data class representing a chunk of text from a PDF."""
    text: str
    page_index: int
    char_start: int
    char_end: int
    embedding: Optional[List[float]] = None
    tokens: Optional[int] = None


class PDFDocument:
    """Class representing a PDF document with extracted text and chunks."""
    
    def __init__(self, file_path: str):
        """
        Initialize a PDFDocument object.
        
        Args:
            file_path: Path to the PDF file
        """
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.pages = []
        self.page_texts = []
        self.chunks = []
        
    async def process(self, 
                      openai_client: AsyncOpenAI,
                      target_chunk_tokens: int = 350, 
                      chunk_overlap: float = 0.3,
                      embedding_model: str = "text-embedding-3-small",
                      llm_usage: Optional[ModelUsageAsync] = None) -> List[PDFChunk]:
        """
        Process the PDF: extract text, chunk, and embed.
        
        Args:
            openai_client: OpenAI client for generating embeddings
            target_chunk_tokens: Target number of tokens per chunk
            chunk_overlap: Overlap between chunks as a fraction
            embedding_model: Name of the embedding model to use
            llm_usage: Optional ModelUsageAsync object for tracking token usage
            
        Returns:
            List of processed PDFChunk objects
        """
        self._extract_text()
        self._create_chunks(target_chunk_tokens, chunk_overlap)
        await self._embed_chunks(openai_client, embedding_model, llm_usage)
        return self.chunks
        
    def _extract_text(self) -> None:
        """Extract text from PDF using pdfplumber."""
        with pdfplumber.open(self.file_path) as pdf:
            self.pages = pdf.pages
            self.page_texts = []
            
            for page in self.pages:
                text = page.extract_text()
                if text:
                    self.page_texts.append(text)
                else:
                    self.page_texts.append("")  # Empty page
    
    def _create_chunks(self, target_chunk_tokens: int = 350, chunk_overlap: float = 0.3) -> None:
        """
        Create chunks from extracted text with target token size and overlap.
        
        Args:
            target_chunk_tokens: Target number of tokens per chunk
            chunk_overlap: Overlap between chunks as a fraction
        """
        # Get encoder for token counting
        encoder = tiktoken.encoding_for_model("gpt-4")
        self.chunks = []
        
        for page_idx, page_text in enumerate(self.page_texts):
            if not page_text.strip():
                continue
                
            # Calculate target characters based on average characters per token
            tokens = encoder.encode(page_text)
            chars_per_token = len(page_text) / len(tokens) if tokens else 5  # Default if page is empty
            target_chars = target_chunk_tokens * chars_per_token
            overlap_chars = int(target_chars * chunk_overlap)
            
            # Split page text into paragraphs using double newlines
            paragraphs = [p for p in re.split(r'\n\s*\n', page_text) if p.strip()]

            # Fallback: if splitting didn't work well (e.g. one huge block), try splitting by sentence
            if len(paragraphs) <= 1:
                paragraphs = re.split(r'(?<=[.?!])\s+', page_text.strip())

                # Final fallback: if sentence split also yields one long block, chunk by fixed character size
                if len(paragraphs) <= 1:
                    paragraph_len = 600  # ~100–150 tokens
                    paragraphs = [
                        page_text[i:i + paragraph_len]
                        for i in range(0, len(page_text), paragraph_len)
                    ]
            
            # Initialize chunk variables
            current_chunk_text = ""
            current_chunk_start = 0
            
            # Process paragraphs into adaptive chunks
            for para in paragraphs:
                para_with_newlines = para + "\n\n"
                
                # If adding this paragraph exceeds target size, create a new chunk
                if len(current_chunk_text) + len(para_with_newlines) > target_chars and current_chunk_text:
                    # Create chunk
                    chunk = PDFChunk(
                        text=current_chunk_text.strip(),
                        page_index=page_idx,
                        char_start=current_chunk_start,
                        char_end=current_chunk_start + len(current_chunk_text)
                    )
                    
                    # Count tokens
                    chunk.tokens = len(encoder.encode(chunk.text))
                    
                    self.chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    overlap_start = max(0, len(current_chunk_text) - overlap_chars)
                    current_chunk_text = current_chunk_text[overlap_start:]
                    current_chunk_start += overlap_start
                
                # Add paragraph to current chunk
                current_chunk_text += para_with_newlines
            
            # Add the last chunk if it's not empty
            if current_chunk_text.strip():
                chunk = PDFChunk(
                    text=current_chunk_text.strip(),
                    page_index=page_idx,
                    char_start=current_chunk_start,
                    char_end=current_chunk_start + len(current_chunk_text)
                )
                
                # Count tokens
                chunk.tokens = len(encoder.encode(chunk.text))
                
                self.chunks.append(chunk)
    
    async def _embed_chunks(self, 
                           openai_client: AsyncOpenAI,
                           embedding_model: str = "text-embedding-3-small",
                           llm_usage: Optional[ModelUsageAsync] = None) -> None:
        """
        Embed chunks using OpenAI's embedding model.
        
        Args:
            openai_client: OpenAI client for generating embeddings
            embedding_model: Name of the embedding model to use
            llm_usage: Optional ModelUsageAsync object for tracking token usage
        """
        # Process in batches to avoid rate limits
        batch_size = 20
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i+batch_size]
            texts = [chunk.text for chunk in batch]
            
            try:
                # Get embeddings from OpenAI
                response = await openai_client.embeddings.create(
                    input=texts,
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
                        label=f"embed_pdf_chunks_{i}-{i+len(batch)}"
                    )
                
                # Assign embeddings to chunks
                for j, embedding_data in enumerate(response.data):
                    self.chunks[i+j].embedding = embedding_data.embedding
                    
            except Exception as e:
                print(f"Error embedding chunks {i} to {i+len(batch)}: {e}")


async def ingest_pdf(
    pdf_path: str,
    openai_client: AsyncOpenAI,
    target_chunk_tokens: int = 350,
    chunk_overlap: float = 0.3,
    embedding_model: str = "text-embedding-3-small",
    llm_usage: Optional[ModelUsageAsync] = None
) -> Tuple[PDFDocument, List[PDFChunk]]:
    """
    Ingest a PDF file: extract text, chunk, and embed.
    
    Args:
        pdf_path: Path to the PDF file
        openai_client: OpenAI client
        target_chunk_tokens: Target token size for chunks
        chunk_overlap: Overlap between chunks (0.0 to 1.0)
        embedding_model: Model to use for embeddings
        llm_usage: Optional tracker for token usage
        
    Returns:
        Tuple of (PDFDocument, List[PDFChunk])
    """
    pdf_doc = PDFDocument(pdf_path)
    chunks = await pdf_doc.process(
        openai_client=openai_client,
        target_chunk_tokens=target_chunk_tokens,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        llm_usage=llm_usage
    )
    
    return pdf_doc, chunks
