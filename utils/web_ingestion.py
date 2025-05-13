"""
Web Ingestion Module

This module provides utilities to:
1. Render web pages with Playwright (handling JS)
2. Strip boilerplate content using readability.js
3. Chunk text with adaptive sizes (~350 tokens) and overlap (30%)
4. Embed chunks using OpenAI's text-embedding-3-small
5. Store chunks with metadata including CSS/XPath selectors
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from openai import AsyncOpenAI
import tiktoken
from dataclasses import dataclass
from .model_costs import ModelUsageAsync
from playwright.async_api import async_playwright
from readability import Document as ReadabilityDocument
import json
import requests
from bs4 import BeautifulSoup


@dataclass
class WebChunk:
    """Data class representing a chunk of text from a web page."""
    text: str
    url: str
    xpath: Optional[str] = None
    css_selector: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
    embedding: Optional[List[float]] = None
    tokens: Optional[int] = None


class WebDocument:
    """Class representing a web document with extracted text and chunks."""
    
    def __init__(self, url: str):
        """
        Initialize a WebDocument object.
        
        Args:
            url: URL of the web page
        """
        self.url = url
        self.title = ""
        self.html = ""
        self.readable_html = ""
        self.text = ""
        self.dom_map = {}  # Maps text ranges to DOM selectors
        self.chunks = []
        
    async def process(self, 
                     openai_client: AsyncOpenAI,
                     target_chunk_tokens: int = 350, 
                     chunk_overlap: float = 0.3,
                     embedding_model: str = "text-embedding-3-small",
                     llm_usage: Optional[ModelUsageAsync] = None) -> List[WebChunk]:
        """
        Process the web page: fetch, extract text, chunk, and embed.
        
        Args:
            openai_client: OpenAI client for generating embeddings
            target_chunk_tokens: Target number of tokens per chunk
            chunk_overlap: Overlap between chunks as a fraction
            embedding_model: Name of the embedding model to use
            llm_usage: Optional ModelUsageAsync object for tracking token usage
            
        Returns:
            List of processed WebChunk objects
        """
        await self._fetch_and_process_url()
        self._create_chunks(target_chunk_tokens, chunk_overlap)
        await self._embed_chunks(openai_client, embedding_model, llm_usage)
        return self.chunks
        
    async def _fetch_and_process_url(self) -> None:
        """Fetch web page content using Playwright and extract text with readability-lxml"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            
            try:
                # Navigate to URL
                await page.goto(self.url, wait_until="networkidle")
                
                # Get full HTML
                self.html = await page.content()
                
                # Get page title
                self.title = await page.title()
                
                # Process HTML with readability-lxml (server-side)
                doc = ReadabilityDocument(self.html)
                self.readable_html = doc.summary()
                
                # Use BeautifulSoup to extract text content and preserve some structure
                soup = BeautifulSoup(self.readable_html, 'html.parser')
                
                # Get clean text with paragraph breaks preserved
                text_parts = []
                for elem in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                    text_parts.append(elem.get_text())
                
                self.text = '\n\n'.join(text_parts)
                
                # Create a map of text ranges to DOM selectors
                self.dom_map = []
                
                # Extract CSS selectors and XPaths using Playwright
                selectors_data = await page.evaluate("""
                    () => {
                        const textMap = [];
                        const elements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li');
                        
                        elements.forEach(element => {
                            const text = element.textContent.trim();
                            if (text) {
                                // Generate XPath
                                let xpath = '';
                                let elem = element;
                                for (; elem && elem.nodeType === 1; elem = elem.parentNode) {
                                    let idx = 0;
                                    let sibling = elem.previousSibling;
                                    while (sibling) {
                                        if (sibling.nodeType === 1 && sibling.nodeName === elem.nodeName) {
                                            idx++;
                                        }
                                        sibling = sibling.previousSibling;
                                    }
                                    const xname = elem.nodeName.toLowerCase();
                                    const pathIndex = (idx ? "[" + (idx+1) + "]" : "");
                                    xpath = "/" + xname + pathIndex + xpath;
                                }
                                
                                // Generate CSS selector
                                let cssPath = [];
                                elem = element;
                                while (elem && elem.nodeType === 1) {
                                    let selector = elem.nodeName.toLowerCase();
                                    if (elem.id) {
                                        selector += '#' + elem.id;
                                        cssPath.unshift(selector);
                                        break;
                                    } else if (elem.className) {
                                        selector += '.' + elem.className.split(' ').join('.');
                                    }
                                    cssPath.unshift(selector);
                                    elem = elem.parentNode;
                                }
                                
                                textMap.push({
                                    text: text,
                                    xpath: xpath,
                                    cssSelector: cssPath.join(' > ')
                                });
                            }
                        });
                        
                        return textMap;
                    }""")
                
                self.dom_map = selectors_data
                
            finally:
                await browser.close()
    
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
        
        if not self.text.strip():
            return
            
        # Calculate target characters based on average characters per token
        tokens = encoder.encode(self.text)
        chars_per_token = len(self.text) / len(tokens) if tokens else 5  # Default if text is empty
        target_chars = target_chunk_tokens * chars_per_token
        overlap_chars = int(target_chars * chunk_overlap)
        
        # Split text into paragraphs using double newlines
        paragraphs = [p for p in re.split(r'\n\s*\n', self.text) if p.strip()]

        # Fallback: if splitting didn't work well, try splitting by sentence
        if len(paragraphs) <= 1:
            paragraphs = re.split(r'(?<=[.?!])\s+', self.text.strip())

            # Final fallback: if sentence split also yields one long block, chunk by fixed character size
            if len(paragraphs) <= 1:
                paragraph_len = 600  # ~100–150 tokens
                paragraphs = [
                    self.text[i:i + paragraph_len]
                    for i in range(0, len(self.text), paragraph_len)
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
                chunk = WebChunk(
                    text=current_chunk_text.strip(),
                    url=self.url,
                    char_start=current_chunk_start,
                    char_end=current_chunk_start + len(current_chunk_text)
                )
                
                # Find best matching DOM selector for this chunk
                self._assign_selector_to_chunk(chunk)
                
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
            chunk = WebChunk(
                text=current_chunk_text.strip(),
                url=self.url,
                char_start=current_chunk_start,
                char_end=current_chunk_start + len(current_chunk_text)
            )
            
            # Find best matching DOM selector for this chunk
            self._assign_selector_to_chunk(chunk)
            
            # Count tokens
            chunk.tokens = len(encoder.encode(chunk.text))
            
            self.chunks.append(chunk)
    
    def _assign_selector_to_chunk(self, chunk: WebChunk) -> None:
        """
        Find the best CSS selector and XPath for a chunk based on text content overlap.
        
        Args:
            chunk: WebChunk to assign selectors to
        """
        # Simple heuristic: find the DOM element that contains the most overlap with the chunk
        best_match = None
        best_overlap = 0
        
        for entry in self.dom_map:
            entry_text = entry.get("text", "")
            if entry_text in chunk.text:
                overlap = len(entry_text)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_match = entry
        
        # If we found a good match, assign its selectors
        if best_match:
            chunk.xpath = best_match.get("xpath")
            chunk.css_selector = best_match.get("cssSelector")
    
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
                        label=f"embed_web_chunks"
                    )
                
                # Assign embeddings to chunks
                for j, embedding_data in enumerate(response.data):
                    if i + j < len(self.chunks):
                        self.chunks[i + j].embedding = embedding_data.embedding
                        
            except Exception as e:
                print(f"Error embedding chunks {i} to {i+batch_size}: {e}")


async def ingest_web_url(
    url: str,
    openai_client: AsyncOpenAI,
    target_chunk_tokens: int = 350,
    chunk_overlap: float = 0.3,
    embedding_model: str = "text-embedding-3-small",
    llm_usage: Optional[ModelUsageAsync] = None
) -> Tuple[WebDocument, List[WebChunk]]:
    """
    Ingest a web URL: fetch content, extract text, chunk, and embed.
    
    Args:
        url: URL to process
        openai_client: OpenAI client
        target_chunk_tokens: Target token size for chunks
        chunk_overlap: Overlap between chunks (0.0 to 1.0)
        embedding_model: Model to use for embeddings
        llm_usage: Optional tracker for token usage
        
    Returns:
        Tuple of (WebDocument, List[WebChunk])
    """
    document = WebDocument(url)
    chunks = await document.process(
        openai_client, 
        target_chunk_tokens,
        chunk_overlap,
        embedding_model,
        llm_usage
    )
    
    print(f"Processed URL: {url}")
    print(f"Extracted {len(chunks)} chunks")
    
    return document, chunks
