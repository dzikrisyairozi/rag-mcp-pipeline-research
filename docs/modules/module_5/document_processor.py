"""
Document Processing for RAG Systems

This module provides classes and functions for document ingestion, processing, and chunking
in RAG applications. It includes document loaders, preprocessors, and chunking strategies.
"""

import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import time


class DocumentType(Enum):
    """Types of documents supported by the document processor."""
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    CSV = "csv"
    JSON = "json"
    DOCX = "docx"
    PPTX = "pptx"
    CODE = "code"


@dataclass
class Document:
    """Represents a document in the RAG system."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_type: DocumentType = DocumentType.TEXT
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the document."""
        content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        return f"doc_{content_hash}"


@dataclass
class DocumentChunk:
    """Represents a chunk of a document."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    chunk_index: int = 0
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the chunk."""
        if self.doc_id:
            return f"{self.doc_id}_chunk_{self.chunk_index}"
        content_hash = hashlib.md5(self.content.encode('utf-8')).hexdigest()
        return f"chunk_{content_hash}"


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """
        Load documents from a source.
        
        Args:
            source: The source to load documents from (e.g., file path, URL)
            
        Returns:
            List of loaded documents
        """
        pass


class TextFileLoader(DocumentLoader):
    """Loader for plain text files."""
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a text file."""
        try:
            print(f"Loading text file from {source}")
            # In a real implementation, this would read the file
            content = f"Mock content loaded from text file {source}"
            
            return [Document(
                id="",
                content=content,
                metadata={
                    "source": source,
                    "file_type": "text",
                },
                doc_type=DocumentType.TEXT
            )]
        except Exception as e:
            print(f"Error loading text file: {e}")
            return []


class PDFLoader(DocumentLoader):
    """Loader for PDF files."""
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a PDF file."""
        try:
            print(f"Loading PDF file from {source}")
            # In a real implementation, this would use a PDF parsing library
            content = f"Mock content loaded from PDF file {source}"
            
            return [Document(
                id="",
                content=content,
                metadata={
                    "source": source,
                    "file_type": "pdf",
                },
                doc_type=DocumentType.PDF
            )]
        except Exception as e:
            print(f"Error loading PDF file: {e}")
            return []


class WebLoader(DocumentLoader):
    """Loader for web pages."""
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a web page."""
        try:
            print(f"Loading web page from {source}")
            # In a real implementation, this would use a web scraping library
            content = f"Mock content loaded from web page {source}"
            
            return [Document(
                id="",
                content=content,
                metadata={
                    "source": source,
                    "file_type": "html",
                },
                doc_type=DocumentType.HTML
            )]
        except Exception as e:
            print(f"Error loading web page: {e}")
            return []


class DocumentPreprocessor(ABC):
    """Abstract base class for document preprocessors."""
    
    @abstractmethod
    def preprocess(self, document: Document) -> Document:
        """
        Preprocess a document.
        
        Args:
            document: The document to preprocess
            
        Returns:
            Preprocessed document
        """
        pass


class HTMLCleaner(DocumentPreprocessor):
    """Preprocessor for cleaning HTML documents."""
    
    def preprocess(self, document: Document) -> Document:
        """Remove HTML tags and clean the document content."""
        if document.doc_type != DocumentType.HTML:
            return document
        
        print(f"Cleaning HTML for document {document.id}")
        # In a real implementation, this would use a proper HTML parsing library
        content = re.sub(r'<[^>]+>', '', document.content)
        
        return Document(
            id=document.id,
            content=content,
            metadata=document.metadata,
            doc_type=document.doc_type
        )


class TextNormalizer(DocumentPreprocessor):
    """Preprocessor for normalizing text."""
    
    def preprocess(self, document: Document) -> Document:
        """Normalize text by removing excess whitespace and standardizing case."""
        print(f"Normalizing text for document {document.id}")
        
        # Remove excess whitespace
        content = re.sub(r'\s+', ' ', document.content).strip()
        
        # In a real implementation, this might include more text normalization steps
        
        return Document(
            id=document.id,
            content=content,
            metadata=document.metadata,
            doc_type=document.doc_type
        )


class MetadataExtractor(DocumentPreprocessor):
    """Preprocessor for extracting metadata from documents."""
    
    def preprocess(self, document: Document) -> Document:
        """Extract metadata from document content."""
        print(f"Extracting metadata for document {document.id}")
        
        # Example metadata extraction (very simplified)
        metadata = document.metadata.copy()
        
        # Extract title (first line or first sentence)
        lines = document.content.split('\n')
        if lines:
            metadata['title'] = lines[0].strip()
        
        # Extract approximate word count
        words = document.content.split()
        metadata['word_count'] = len(words)
        
        # In a real implementation, this would extract more useful metadata
        
        return Document(
            id=document.id,
            content=document.content,
            metadata=metadata,
            doc_type=document.doc_type
        )


class ChunkingStrategy(ABC):
    """Abstract base class for document chunking strategies."""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """
        Split a document into chunks.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of document chunks
        """
        pass


class FixedSizeChunker(ChunkingStrategy):
    """Chunking strategy that splits documents into fixed-size chunks."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = max(chunk_size, 100)  # Minimum chunk size of 100
        self.chunk_overlap = min(chunk_overlap, self.chunk_size // 2)
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Split a document into fixed-size chunks."""
        print(f"Chunking document {document.id} with fixed size strategy")
        
        # Get document content
        text = document.content
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [DocumentChunk(
                id="",
                content=text,
                metadata=document.metadata.copy(),
                doc_id=document.id,
                chunk_index=0
            )]
        
        # Split text into chunks
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.chunk_size, len(text))
            
            # If not at the end of the text, try to find a good break point
            if end < len(text):
                # Try to find sentence boundary
                sentence_break = text.rfind('. ', start, end)
                if sentence_break > start + self.chunk_size // 2:
                    end = sentence_break + 2  # Include the period and space
                else:
                    # Try to find space
                    space = text.rfind(' ', start, end)
                    if space > start + self.chunk_size // 2:
                        end = space + 1  # Include the space
            
            # Create chunk
            chunk_text = text[start:end]
            
            # Create metadata with position information
            metadata = document.metadata.copy()
            metadata['chunk_index'] = chunk_index
            metadata['start_char'] = start
            metadata['end_char'] = end
            
            chunks.append(DocumentChunk(
                id="",
                content=chunk_text,
                metadata=metadata,
                doc_id=document.id,
                chunk_index=chunk_index
            ))
            
            # Move start position for next chunk, considering overlap
            start = end - self.chunk_overlap
            chunk_index += 1
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """Chunking strategy that splits documents into semantic units."""
    
    def __init__(self, max_chunk_size: int = 800):
        """
        Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters
        """
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, document: Document) -> List[DocumentChunk]:
        """Split a document into semantic chunks (sections, paragraphs)."""
        print(f"Chunking document {document.id} with semantic strategy")
        
        # Get document content
        text = document.content
        
        # If text is smaller than max chunk size, return as single chunk
        if len(text) <= self.max_chunk_size:
            return [DocumentChunk(
                id="",
                content=text,
                metadata=document.metadata.copy(),
                doc_id=document.id,
                chunk_index=0
            )]
        
        # Split by section headers or paragraphs
        # This is a simplified implementation
        sections = self._split_by_sections(text)
        
        # Process each section
        chunks = []
        chunk_index = 0
        
        for section in sections:
            # If section is too large, split it further
            if len(section) > self.max_chunk_size:
                # Split by paragraphs
                paragraphs = self._split_by_paragraphs(section)
                
                # Create chunks from paragraphs
                current_chunk = ""
                
                for paragraph in paragraphs:
                    # If adding this paragraph would exceed max size, create a new chunk
                    if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                        # Create chunk from current text
                        metadata = document.metadata.copy()
                        metadata['chunk_index'] = chunk_index
                        
                        chunks.append(DocumentChunk(
                            id="",
                            content=current_chunk.strip(),
                            metadata=metadata,
                            doc_id=document.id,
                            chunk_index=chunk_index
                        ))
                        
                        chunk_index += 1
                        current_chunk = paragraph
                    else:
                        # Add paragraph to current chunk
                        if current_chunk:
                            current_chunk += "\n\n"
                        current_chunk += paragraph
                
                # Add the last chunk if there's any text left
                if current_chunk:
                    metadata = document.metadata.copy()
                    metadata['chunk_index'] = chunk_index
                    
                    chunks.append(DocumentChunk(
                        id="",
                        content=current_chunk.strip(),
                        metadata=metadata,
                        doc_id=document.id,
                        chunk_index=chunk_index
                    ))
                    
                    chunk_index += 1
            else:
                # Add section as a chunk
                metadata = document.metadata.copy()
                metadata['chunk_index'] = chunk_index
                
                chunks.append(DocumentChunk(
                    id="",
                    content=section.strip(),
                    metadata=metadata,
                    doc_id=document.id,
                    chunk_index=chunk_index
                ))
                
                chunk_index += 1
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by section headers."""
        # Simple pattern for section headers (e.g., "## Section Title")
        pattern = r'(?m)^#{1,3} '
        
        # Find all section headers
        matches = list(re.finditer(pattern, text))
        
        # If no headers found, return the entire text as one section
        if not matches:
            return [text]
        
        # Split text by section headers
        sections = []
        start_idx = 0
        
        for match in matches:
            if match.start() > start_idx:
                sections.append(text[start_idx:match.start()])
            start_idx = match.start()
        
        # Add the last section
        if start_idx < len(text):
            sections.append(text[start_idx:])
        
        return sections
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs."""
        # Simple split by double newline
        paragraphs = re.split(r'\n\s*\n', text)
        return [p for p in paragraphs if p.strip()]


class DocumentProcessingPipeline:
    """Pipeline for processing documents through multiple stages."""
    
    def __init__(self, 
                loader: DocumentLoader,
                preprocessors: List[DocumentPreprocessor],
                chunking_strategy: ChunkingStrategy):
        """
        Initialize the pipeline.
        
        Args:
            loader: Document loader to use
            preprocessors: List of preprocessors to apply
            chunking_strategy: Strategy for chunking documents
        """
        self.loader = loader
        self.preprocessors = preprocessors
        self.chunking_strategy = chunking_strategy
    
    def process(self, source: str) -> List[DocumentChunk]:
        """
        Process documents from a source through the pipeline.
        
        Args:
            source: Source to load documents from
            
        Returns:
            List of processed document chunks
        """
        # Step 1: Load documents
        documents = self.loader.load(source)
        print(f"Loaded {len(documents)} documents from {source}")
        
        # Step 2: Preprocess documents
        processed_docs = []
        for doc in documents:
            for preprocessor in self.preprocessors:
                doc = preprocessor.preprocess(doc)
            processed_docs.append(doc)
        
        print(f"Preprocessed {len(processed_docs)} documents")
        
        # Step 3: Chunk documents
        all_chunks = []
        for doc in processed_docs:
            chunks = self.chunking_strategy.chunk(doc)
            all_chunks.extend(chunks)
        
        print(f"Created {len(all_chunks)} chunks")
        
        return all_chunks


class DocumentProcessorFactory:
    """Factory for creating document processing pipelines."""
    
    @staticmethod
    def create_pipeline(source_type: str, chunking_method: str = "fixed") -> DocumentProcessingPipeline:
        """
        Create a document processing pipeline based on source type and chunking method.
        
        Args:
            source_type: Type of source ("text", "pdf", "web")
            chunking_method: Method for chunking ("fixed", "semantic")
            
        Returns:
            Configured document processing pipeline
        """
        # Select loader based on source type
        if source_type == "pdf":
            loader = PDFLoader()
        elif source_type == "web":
            loader = WebLoader()
        else:
            # Default to text
            loader = TextFileLoader()
        
        # Create preprocessors (common for all types)
        preprocessors = [TextNormalizer(), MetadataExtractor()]
        
        # Add type-specific preprocessors
        if source_type == "web":
            preprocessors.insert(0, HTMLCleaner())
        
        # Select chunking strategy
        if chunking_method == "semantic":
            chunking_strategy = SemanticChunker()
        else:
            # Default to fixed
            chunking_strategy = FixedSizeChunker()
        
        # Create pipeline
        return DocumentProcessingPipeline(loader, preprocessors, chunking_strategy)


def process_documents(sources: List[str], 
                     source_type: str = "text",
                     chunking_method: str = "fixed") -> List[DocumentChunk]:
    """
    Process multiple documents from sources.
    
    Args:
        sources: List of sources to process
        source_type: Type of the sources
        chunking_method: Method for chunking
        
    Returns:
        List of all document chunks
    """
    pipeline = DocumentProcessorFactory.create_pipeline(source_type, chunking_method)
    
    all_chunks = []
    for source in sources:
        chunks = pipeline.process(source)
        all_chunks.extend(chunks)
    
    return all_chunks


if __name__ == "__main__":
    print("Document Processing for RAG Systems")
    print("----------------------------------")
    
    # Example usage
    sources = ["example.txt", "sample.pdf", "https://example.com"]
    
    # Process documents with different configurations
    print("\nProcessing text files with fixed-size chunking:")
    text_chunks = process_documents(sources[:1], "text", "fixed")
    print(f"Generated {len(text_chunks)} chunks")
    
    print("\nProcessing PDF files with semantic chunking:")
    pdf_chunks = process_documents(sources[1:2], "pdf", "semantic")
    print(f"Generated {len(pdf_chunks)} chunks")
    
    print("\nProcessing web pages:")
    web_chunks = process_documents(sources[2:], "web", "semantic")
    print(f"Generated {len(web_chunks)} chunks")
    
    # Demonstrate custom pipeline
    print("\nDemonstrating custom pipeline:")
    loader = WebLoader()
    preprocessors = [HTMLCleaner(), TextNormalizer(), MetadataExtractor()]
    chunker = FixedSizeChunker(chunk_size=300, chunk_overlap=100)
    
    pipeline = DocumentProcessingPipeline(loader, preprocessors, chunker)
    custom_chunks = pipeline.process("https://example.com/custom")
    
    # Print sample chunk
    if custom_chunks:
        print("\nSample chunk:")
        print(f"ID: {custom_chunks[0].id}")
        print(f"Content: {custom_chunks[0].content[:100]}...")
        print(f"Metadata: {custom_chunks[0].metadata}")
        
    print("\nNote: This is a simplified implementation. In a real-world RAG system,")
    print("document processing would include more advanced features like:")
    print("- OCR for scanned documents")
    print("- Format-specific extractors (tables, images, etc.)")
    print("- Language detection and translation")
    print("- Entity recognition and enrichment")
    print("- Content filtering and quality assessment") 