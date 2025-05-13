"""
PDF Highlighter Module

This module provides PDF viewing capabilities with citation highlighting using PyMuPDF and Tkinter.
"""

import os
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from typing import List, Dict, Any, Tuple, Optional, Union
import fitz  # PyMuPDF
import threading
from PIL import Image, ImageTk


class PDFHighlighter:
    """
    PDF viewer with citation highlighting capabilities using PyMuPDF and Tkinter.
    Allows highlighting of specific text passages across multiple pages.
    """
    
    def __init__(self, master: tk.Tk = None, pdf_path: str = None, title: str = "PDF Citation Viewer", answer: str = None):
        """
        Initialize the PDF highlighter.
        
        Args:
            master: Tkinter root window or None to create a new one
            pdf_path: Path to the PDF file to display
            title: Window title
        """
        # Create Tkinter root if not provided
        self.master = master or tk.Tk()
        self.master.title(title)
        self.master.geometry("1200x800")
        
        # PDF document
        self.pdf_path = None
        self.doc = None
        self.current_page = 0
        self.total_pages = 0
        self.zoom = 1.0
        self.rotation = 0
        self.citations = []
        self.page_images = []  # Cache for rendered page images
        self.answer = answer  # Store the answer to display
        
        # UI components
        self._create_ui()
        
        # Load PDF if provided
        if pdf_path:
            self.load_pdf(pdf_path)
    
    def _create_ui(self):
        """Create the user interface components."""
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side: PDF viewer
        viewer_frame = ttk.Frame(main_frame)
        viewer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for displaying PDF
        self.canvas_frame = ttk.Frame(viewer_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg='gray80')
        self.scrollbar_y = ttk.Scrollbar(self.canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scrollbar_x = ttk.Scrollbar(self.canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.config(yscrollcommand=self.scrollbar_y.set, xscrollcommand=self.scrollbar_x.set)
        
        self.scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Bind mouse wheel for scrolling
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)
        
        # Controls frame
        controls_frame = ttk.Frame(viewer_frame)
        controls_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Navigation buttons
        ttk.Button(controls_frame, text="Previous", command=self.previous_page).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Next", command=self.next_page).pack(side=tk.LEFT, padx=2)
        
        # Page indicator
        self.page_label = ttk.Label(controls_frame, text="Page: 0 / 0")
        self.page_label.pack(side=tk.LEFT, padx=20)
        
        # Zoom controls
        ttk.Button(controls_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(controls_frame, text="Reset Zoom", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        
        # Right side: Answer and Citations panel
        right_frame = ttk.Frame(main_frame, width=350)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5, pady=5)
        right_frame.pack_propagate(False)  # Prevent frame from shrinking
        
        # Answer section
        ttk.Label(right_frame, text="Answer", font=('Arial', 12, 'bold')).pack(anchor=tk.W, padx=5, pady=5)
        
        # Answer text box
        answer_container = ttk.Frame(right_frame, height=150)
        answer_container.pack(fill=tk.X, padx=5, pady=5)
        answer_container.pack_propagate(False)  # Prevent frame from shrinking
        
        self.answer_text = scrolledtext.ScrolledText(answer_container, wrap=tk.WORD, width=30, height=10)
        self.answer_text.pack(fill=tk.BOTH, expand=True)
        
        # Insert answer if provided
        if self.answer:
            self.answer_text.insert(tk.END, self.answer)
        else:
            self.answer_text.insert(tk.END, "No answer provided.")
        self.answer_text.config(state=tk.DISABLED)
        
        # Separator
        ttk.Separator(right_frame, orient='horizontal').pack(fill=tk.X, padx=5, pady=10)
        
        # Citations section
        ttk.Label(right_frame, text="Citations", font=('Arial', 12, 'bold')).pack(anchor=tk.W, padx=5, pady=5)
        
        # Citations list with scrollbar
        citations_container = ttk.Frame(right_frame)
        citations_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.citations_text = scrolledtext.ScrolledText(citations_container, wrap=tk.WORD, width=30, height=20)
        self.citations_text.pack(fill=tk.BOTH, expand=True)
        self.citations_text.config(state=tk.DISABLED)
        
        # Help text
        help_frame = ttk.Frame(right_frame)
        help_frame.pack(fill=tk.X, padx=5, pady=5)
        
        help_text = "Click on a citation to jump to that page. Citations are highlighted in yellow in the document."
        ttk.Label(help_frame, text=help_text, wraplength=280).pack(fill=tk.X)
    
    def load_pdf(self, pdf_path: str) -> bool:
        """
        Load a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            True if PDF was loaded successfully, False otherwise
        """
        try:
            # Close current document if open
            if self.doc:
                self.doc.close()
            
            # Check if file exists
            if not os.path.exists(pdf_path):
                messagebox.showerror("Error", f"File not found: {pdf_path}")
                return False
            
            # Open PDF
            self.pdf_path = pdf_path
            self.doc = fitz.open(pdf_path)
            self.total_pages = len(self.doc)
            self.current_page = 0
            self.page_images = [None] * self.total_pages  # Reset page cache
            
            # Update page label
            self._update_page_label()
            
            # Render first page
            self.render_page(0)
            
            return True
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load PDF: {str(e)}")
            return False
    
    def render_page(self, page_num: int) -> None:
        """
        Render a specific page of the PDF.
        
        Args:
            page_num: Page number to render (0-indexed)
        """
        if not self.doc or page_num < 0 or page_num >= self.total_pages:
            return
        
        self.current_page = page_num
        self._update_page_label()
        
        # Get the page
        page = self.doc[page_num]
        
        # Clear current content
        self.canvas.delete("all")
        
        # Check if page is already rendered at current zoom
        cache_key = (page_num, self.zoom, self.rotation)
        if page_num < len(self.page_images) and self.page_images[page_num] and \
           self.page_images[page_num][0] == cache_key:
            # Use cached image
            img = self.page_images[page_num][1]
        else:
            # Render page to pixmap with appropriate zoom
            matrix = fitz.Matrix(self.zoom, self.zoom).prerotate(self.rotation)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            
            # Convert to PIL Image and then to PhotoImage
            img_data = pix.samples
            img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
            img = ImageTk.PhotoImage(img)
            
            # Cache the image
            self.page_images[page_num] = (cache_key, img)
        
        # Add image to canvas
        img_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # Keep a reference to prevent garbage collection
        
        # Configure canvas scrollregion
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        
        # Highlight citations on this page
        self._highlight_citations(page_num)
    
    def _highlight_citations(self, page_num: int) -> None:
        """
        Highlight citations on the current page.
        
        Args:
            page_num: Page number (0-indexed)
        """
        if not self.doc or not self.citations:
            return
        
        # Get page
        page = self.doc[page_num]
        
        # Find citations for this page
        # Handle both string and int page numbers
        page_citations = []
        for cite in self.citations:
            try:
                cite_page = int(cite.get('page', 0))
                if cite_page - 1 == page_num:
                    page_citations.append(cite)
            except (ValueError, TypeError):
                # Log invalid page number format
                print(f"Warning: Invalid page number format: {cite.get('page')}")
                continue
        print(f"Found {len(page_citations)} citations for page {page_num+1}")
            
        # Debug info about page numbers
        if page_citations:
            citation_page_nums = [cite.get('page') for cite in page_citations]
            print(f"Citation page numbers: {citation_page_nums}")
        
        # Get all text on the page for more flexible matching
        try:
            # Extract page text with layout preservation
            page_text = page.get_text("text")
            print(f"Page {page_num+1} has {len(page_text)} characters of text")
        except Exception as e:
            print(f"Error extracting text from page {page_num+1}: {e}")
            page_text = ""
        
        for citation in page_citations:
            text = citation.get('text', '').strip()
            if not text:
                continue
            
            print(f"Looking for citation: {text[:50]}...")
                
            # Multiple approaches to find and highlight text
            highlight_methods = [
                self._highlight_with_exact_match,
                self._highlight_with_simplified_match,
                self._highlight_with_fuzzy_match,
                self._highlight_with_word_search
            ]
            
            # Try each method until one succeeds
            highlighted = False
            for method in highlight_methods:
                if method(page, text, page_num):
                    highlighted = True
                    print(f"Successfully highlighted using {method.__name__}")
                    break
            
            if not highlighted:
                print(f"WARNING: Failed to highlight citation on page {page_num+1}")
                # As a last resort, add a citation marker at the top of the page
                self._add_citation_marker(page_num, text)
    
    def _highlight_with_exact_match(self, page, text, page_num):
        """Try to highlight using exact text match."""
        text_instances = page.search_for(text.strip())
        
        if text_instances:
            self._create_highlights(text_instances, page_num)
            return True
        return False
    
    def _highlight_with_simplified_match(self, page, text, page_num):
        """Try to highlight using simplified text (no extra whitespace)."""
        # Clean the search text - normalize whitespace
        simplified_text = ' '.join(text.split())
        
        # Try with different segments if text is long
        if len(simplified_text) > 40:
            segments = [
                simplified_text[:100],
                simplified_text[:50],
                ' '.join(simplified_text.split()[:10]),  # First 10 words
                ' '.join(simplified_text.split()[:5])    # First 5 words
            ]
            
            for segment in segments:
                if segment and len(segment) > 4:  # Only try if segment has substance
                    text_instances = page.search_for(segment.strip())
                    if text_instances:
                        self._create_highlights(text_instances, page_num)
                        return True
        else:
            # For shorter text, try direct match
            text_instances = page.search_for(simplified_text)
            if text_instances:
                self._create_highlights(text_instances, page_num)
                return True
        
        return False
    
    def _highlight_with_fuzzy_match(self, page, text, page_num):
        """Try to highlight using the first and last few words."""
        words = text.split()
        
        if len(words) >= 5:
            # Try first few words
            start_text = ' '.join(words[:4])
            text_instances = page.search_for(start_text)
            
            if text_instances:
                self._create_highlights(text_instances, page_num)
                return True
                
            # Try last few words
            end_text = ' '.join(words[-4:])
            text_instances = page.search_for(end_text)
            
            if text_instances:
                self._create_highlights(text_instances, page_num)
                return True
        
        return False
    
    def _highlight_with_word_search(self, page, text, page_num):
        """Try to highlight using distinctive words or phrases in the text."""
        # Find distinctive words (longer words are more distinctive)
        words = [w for w in text.split() if len(w) > 5]
        phrases = []
        
        # Create 2-3 word phrases for search
        if len(words) >= 3:
            for i in range(len(words) - 2):
                phrases.append(f"{words[i]} {words[i+1]}")
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        # Try phrases first (more specific)
        for phrase in phrases:
            if len(phrase) > 7:  # Make sure phrase has substance
                text_instances = page.search_for(phrase)
                if text_instances:
                    self._create_highlights(text_instances, page_num)
                    return True
        
        # Then try individual distinctive words
        for word in words:
            if len(word) > 7:  # Only use very distinctive words to avoid false matches
                text_instances = page.search_for(word)
                if text_instances:
                    self._create_highlights(text_instances, page_num)
                    return True
        
        return False
    
    def _create_highlights(self, instances, page_num):
        """Create highlight rectangles for text instances."""
        for inst in instances:
            # Convert PDF coordinates to canvas coordinates (applying zoom)
            x0, y0, x1, y1 = inst
            x0 *= self.zoom
            y0 *= self.zoom
            x1 *= self.zoom
            y1 *= self.zoom
            
            # Make the highlight slightly larger for better visibility
            padding = 2 * self.zoom
            x0 -= padding
            y0 -= padding
            x1 += padding
            y1 += padding
            
            # FIX: First, get all canvas items at this position to determine image item
            overlapping_items = self.canvas.find_overlapping(x0, y0, x1, y1)
            image_item = None
            for item_id in overlapping_items:
                if self.canvas.type(item_id) == "image":
                    image_item = item_id
                    break
            
            # Create highlight rectangle with genuinely transparent yellow color
            # Use the most transparent built-in bitmap pattern
            rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1, 
                fill='#FFFF99',  # Very light yellow
                stipple='gray12',  # Standard stipple pattern with good transparency
                outline='#FFA500',  # Orange outline
                width=1,  # Thinner border for less visual weight
                tags=f"highlight_{page_num}"
            )
            
            # IMPORTANT: Raise the highlight above the PDF image
            # but keep text above the highlight
            if image_item:
                # Place highlight just above the image
                self.canvas.tag_raise(rect_id, image_item)
    
    def _add_citation_marker(self, page_num, text):
        """Add a citation marker at the top of the page as fallback."""
        # Create a marker at the top of the page
        marker_height = 30 * self.zoom
        marker_width = self.canvas.winfo_width() if self.canvas.winfo_width() > 0 else 600
        
        # Create a yellow background rectangle
        rect_id = self.canvas.create_rectangle(
            10, 10, marker_width - 10, marker_height,
            fill='#FFFFCC',
            outline='#FFA500',
            width=2,
            tags=f"marker_{page_num}"
        )
        
        # Add text explaining this is a citation
        text_id = self.canvas.create_text(
            20, marker_height / 2,
            text=f"Citation present on this page: {text[:50]}...",
            anchor=tk.W,
            fill='black',
            font=("Arial", 10, "bold"),
            tags=f"marker_{page_num}"
        )
    
    def set_citations(self, citations: List[Dict[str, Any]]) -> None:
        """
        Set the citations to highlight.
        
        Args:
            citations: List of citation objects with 'page' and 'text' fields
        """
        self.citations = citations
        
        # Update citations panel
        self._update_citations_panel()
        
        # Re-render current page to show highlights
        if self.doc:
            self.render_page(self.current_page)
    
    def _update_citations_panel(self) -> None:
        """Update the citations panel with the current citations."""
        # Enable editing
        self.citations_text.config(state=tk.NORMAL)
        
        # Clear current content
        self.citations_text.delete(1.0, tk.END)
        
        # Add each citation
        for i, citation in enumerate(self.citations):
            page = citation.get('page', 0)
            text = citation.get('text', '')
            
            # Limit text length for display
            display_text = text[:100] + '...' if len(text) > 100 else text
            
            # Add citation entry
            entry_text = f"Citation {i+1} (Page {page}):\n{display_text}\n\n"
            
            # Insert with tag for click handling
            start_pos = self.citations_text.index(tk.END)
            self.citations_text.insert(tk.END, entry_text)
            end_pos = self.citations_text.index(tk.END)
            
            # Create tag for this citation
            tag_name = f"citation_{i}"
            self.citations_text.tag_add(tag_name, start_pos, f"{start_pos}+1l")
            self.citations_text.tag_config(tag_name, background='#e0e0e0', foreground='blue')
            
            # Bind click event
            self.citations_text.tag_bind(tag_name, '<Button-1>', 
                                        lambda e, p=page: self.goto_page(p-1))
        
        # Disable editing
        self.citations_text.config(state=tk.DISABLED)
    
    def goto_page(self, page_num: int) -> None:
        """
        Go to a specific page.
        
        Args:
            page_num: Page number (0-indexed)
        """
        if not self.doc or page_num < 0 or page_num >= self.total_pages:
            return
            
        self.render_page(page_num)
    
    def next_page(self) -> None:
        """Go to the next page."""
        if self.current_page < self.total_pages - 1:
            self.render_page(self.current_page + 1)
    
    def previous_page(self) -> None:
        """Go to the previous page."""
        if self.current_page > 0:
            self.render_page(self.current_page - 1)
    
    def zoom_in(self) -> None:
        """Increase zoom level."""
        self.zoom *= 1.2
        self.render_page(self.current_page)
    
    def zoom_out(self) -> None:
        """Decrease zoom level."""
        self.zoom *= 0.8
        self.render_page(self.current_page)
    
    def reset_zoom(self) -> None:
        """Reset zoom to default level."""
        self.zoom = 1.0
        self.render_page(self.current_page)
    
    def _update_page_label(self) -> None:
        """Update the page number label."""
        self.page_label.config(text=f"Page: {self.current_page + 1} / {self.total_pages}")
    
    def _on_mousewheel(self, event) -> None:
        """Handle mousewheel scrolling."""
        # Scroll up/down
        if event.delta:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    def run(self) -> None:
        """Start the main application loop."""
        self.master.mainloop()
    
    def close(self) -> None:
        """Close the PDF document and application."""
        if self.doc:
            self.doc.close()
        
        # Close Tkinter window if it exists and is not destroyed
        if self.master and self.master.winfo_exists():
            self.master.destroy()


def format_citations_for_highlighting(citations: List[Dict[str, Union[str, int]]]) -> List[Dict[str, Any]]:
    """
    Format citation data for the PDF highlighter.
    
    Args:
        citations: List of citation objects from QA system
        
    Returns:
        List of formatted citation objects for highlighting
    """
    # Ensure citations have correct format: [{"page": page_number, "text": "cited text"}, ...]
    formatted_citations = []
    
    for citation in citations:
        # Extract page number
        if "page" in citation:
            # Ensure page is an integer
            try:
                page = int(citation["page"])
            except (ValueError, TypeError):
                continue
        else:
            # Skip citations without page info
            continue
            
        # Extract text
        if "text" in citation and citation["text"]:
            text = citation["text"]
        else:
            # Skip citations without text
            continue
            
        # Add to formatted citations
        formatted_citations.append({
            "page": page,
            "text": text
        })
    
    return formatted_citations


def highlight_pdf_with_citations(pdf_path: str, citations: List[Dict[str, Any]], answer: str = None) -> None:
    """
    Open a PDF with highlighted citations in a Tkinter window.
    
    Args:
        pdf_path: Path to the PDF file
        citations: List of citation objects with page and text fields
        answer: Optional answer text to display in the viewer
    """
    def run_app():
        root = tk.Tk()
        app = PDFHighlighter(root, pdf_path=pdf_path, answer=answer)
        app.set_citations(citations)
        app.run()
    
    # Run in the main thread if we're already in the main thread
    # Otherwise start a new thread
    if threading.current_thread() is threading.main_thread():
        run_app()
    else:
        # Not ideal for production, but works for demonstration
        # In production, consider using multiprocessing
        threading.Thread(target=run_app).start()


# Example usage in a Jupyter notebook
def display_pdf_with_citations_button(pdf_path: str, citations: List[Dict[str, Any]]) -> None:
    """
    Display a button in Jupyter to open PDF with citations.
    
    Args:
        pdf_path: Path to the PDF file
        citations: List of citation objects with page and text fields
    """
    from IPython.display import HTML, display
    
    # Format citations
    formatted_citations = format_citations_for_highlighting(citations)
    
    # Format the button HTML with escaped citations for JavaScript
    import json
    citations_json = json.dumps(formatted_citations).replace('"', '\\"')
    
    button_html = f"""
    <button 
        onclick="
            IPython.notebook.kernel.execute(
                'from utils.pdf_highlighter import highlight_pdf_with_citations; highlight_pdf_with_citations(\"{pdf_path}\", {citations_json})'
            );
        "
        style="background-color: #4CAF50; color: white; padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer;"
    >
        View PDF with Highlights
    </button>
    """
    
    display(HTML(button_html))


if __name__ == "__main__":
    # Example usage when run as script
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_highlighter.py <pdf_path>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Sample citations
    sample_citations = [
        {"page": 1, "text": "Example text from page 1"},
        {"page": 2, "text": "Another example from page 2"}
    ]
    
    # Open PDF with citations
    highlight_pdf_with_citations(pdf_path, sample_citations)
