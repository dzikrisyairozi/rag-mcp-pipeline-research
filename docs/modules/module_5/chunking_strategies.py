"""
Chunking Strategies for RAG Systems

This module provides various strategies for splitting documents into chunks for RAG applications.
Each strategy is optimized for different document types and retrieval requirements.
"""

import re
import nltk
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import hashlib


@dataclass
class TextChunk:
    """Represents a chunk of text."""
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()
    
    def _generate_id(self) -> str:
        """Generate a unique ID for the chunk."""
        content_hash = hashlib.md5(self.text.encode('utf-8')).hexdigest()
        return f"chunk_{content_hash}"


class ChunkingStrategy(ABC):
    """Abstract base class for document chunking strategies."""
    
    @abstractmethod
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """
        Split text into chunks according to the strategy.
        
        Args:
            text: The text to split
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of text chunks
        """
        pass


class CharacterTextSplitter(ChunkingStrategy):
    """Split text based on a maximum number of characters."""
    
    def __init__(self, 
                chunk_size: int = 1000, 
                chunk_overlap: int = 200,
                separator: str = "\n"):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: Number of characters of overlap between chunks
            separator: Character(s) to use as separator for finding split points
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.separator = separator
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Split text into chunks based on character count."""
        if metadata is None:
            metadata = {}
        
        # If text is shorter than chunk size, return it as is
        if len(text) <= self.chunk_size:
            return [TextChunk(id="", text=text, metadata=metadata.copy())]
        
        # Split text into chunks
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Find end position
            end = start + self.chunk_size
            
            # Adjust end position to not exceed text length
            if end > len(text):
                end = len(text)
            else:
                # Try to find a separator to split on
                separator_pos = text.rfind(self.separator, start, end)
                if separator_pos > start:
                    end = separator_pos + len(self.separator)
            
            # Extract chunk text
            chunk_text = text[start:end]
            
            # Create metadata for the chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'start_char': start,
                'end_char': end,
            })
            
            # Create chunk
            chunks.append(TextChunk(
                id="",
                text=chunk_text,
                metadata=chunk_metadata
            ))
            
            # Update start position for next chunk, considering overlap
            start = end - self.chunk_overlap
            chunk_index += 1
        
        return chunks


class TokenTextSplitter(ChunkingStrategy):
    """Split text based on a maximum number of tokens."""
    
    def __init__(self, 
                chunk_size: int = 256, 
                chunk_overlap: int = 20,
                tokenizer: Optional[Any] = None):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum number of tokens per chunk
            chunk_overlap: Number of tokens of overlap between chunks
            tokenizer: Optional custom tokenizer, defaults to simple whitespace tokenization
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        self.tokenizer = tokenizer
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Split text into chunks based on token count."""
        if metadata is None:
            metadata = {}
        
        # Tokenize the text
        if self.tokenizer:
            tokens = self.tokenizer(text)
        else:
            # Simple whitespace tokenization
            tokens = text.split()
        
        # If text has fewer tokens than chunk size, return it as is
        if len(tokens) <= self.chunk_size:
            return [TextChunk(id="", text=text, metadata=metadata.copy())]
        
        # Split tokens into chunks
        token_chunks = []
        start = 0
        
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            token_chunks.append(tokens[start:end])
            start += self.chunk_size - self.chunk_overlap
        
        # Convert token chunks back to text
        chunks = []
        for i, token_chunk in enumerate(token_chunks):
            # Simple space joining for whitespace tokenization
            if self.tokenizer:
                # If a custom tokenizer is used, the detokenization logic would go here
                chunk_text = " ".join(token_chunk)
            else:
                chunk_text = " ".join(token_chunk)
            
            # Create metadata for the chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': i,
                'token_count': len(token_chunk),
            })
            
            # Create chunk
            chunks.append(TextChunk(
                id="",
                text=chunk_text,
                metadata=chunk_metadata
            ))
        
        return chunks


class SentenceTextSplitter(ChunkingStrategy):
    """Split text into chunks of sentences, preserving sentence boundaries."""
    
    def __init__(self, 
                max_sentences: int = 15, 
                overlap_sentences: int = 2):
        """
        Initialize the text splitter.
        
        Args:
            max_sentences: Maximum number of sentences per chunk
            overlap_sentences: Number of sentences to overlap between chunks
        """
        self.max_sentences = max_sentences
        self.overlap_sentences = min(overlap_sentences, max_sentences // 2)
        
        # Try to load NLTK sentence tokenizer, fall back to regex if not available
        try:
            nltk.download('punkt', quiet=True)
            self.nltk_available = True
        except:
            self.nltk_available = False
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Split text into chunks based on sentence boundaries."""
        if metadata is None:
            metadata = {}
        
        # Split text into sentences
        if self.nltk_available:
            try:
                sentences = nltk.sent_tokenize(text)
            except:
                sentences = self._fallback_sentence_split(text)
        else:
            sentences = self._fallback_sentence_split(text)
        
        # If text has fewer sentences than max, return it as is
        if len(sentences) <= self.max_sentences:
            return [TextChunk(id="", text=text, metadata=metadata.copy())]
        
        # Split sentences into chunks
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(sentences):
            # Calculate end position
            end = min(start + self.max_sentences, len(sentences))
            
            # Extract sentences and join into a chunk
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences)
            
            # Create metadata for the chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'sentence_count': len(chunk_sentences),
                'start_sentence': start,
                'end_sentence': end,
            })
            
            # Create chunk
            chunks.append(TextChunk(
                id="",
                text=chunk_text,
                metadata=chunk_metadata
            ))
            
            # Update start position for next chunk, considering overlap
            start = end - self.overlap_sentences
            chunk_index += 1
        
        return chunks
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """
        Fallback sentence splitting using regex when NLTK is not available.
        
        Args:
            text: Text to split into sentences
            
        Returns:
            List of sentences
        """
        # Simple regex for sentence boundaries
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        
        # Handle cases where regex doesn't split correctly
        result = []
        for sentence in sentences:
            # If sentence is very long, it might have missed some boundaries
            if len(sentence) > 300:
                # Try a simpler approach for long sentences
                sub_sentences = re.split(r'[.!?]\s+', sentence)
                # Add punctuation back
                for i, s in enumerate(sub_sentences[:-1]):
                    result.append(s + '.')
                if sub_sentences:
                    result.append(sub_sentences[-1])
            else:
                result.append(sentence)
        
        return result


class ParagraphTextSplitter(ChunkingStrategy):
    """Split text into chunks of paragraphs, preserving paragraph boundaries."""
    
    def __init__(self, 
                max_paragraphs: int = 5, 
                overlap_paragraphs: int = 1):
        """
        Initialize the text splitter.
        
        Args:
            max_paragraphs: Maximum number of paragraphs per chunk
            overlap_paragraphs: Number of paragraphs to overlap between chunks
        """
        self.max_paragraphs = max_paragraphs
        self.overlap_paragraphs = min(overlap_paragraphs, max_paragraphs // 2)
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Split text into chunks based on paragraph boundaries."""
        if metadata is None:
            metadata = {}
        
        # Split text into paragraphs (by double newline)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # If text has fewer paragraphs than max, return it as is
        if len(paragraphs) <= self.max_paragraphs:
            return [TextChunk(id="", text=text, metadata=metadata.copy())]
        
        # Split paragraphs into chunks
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(paragraphs):
            # Calculate end position
            end = min(start + self.max_paragraphs, len(paragraphs))
            
            # Extract paragraphs and join into a chunk
            chunk_paragraphs = paragraphs[start:end]
            chunk_text = "\n\n".join(chunk_paragraphs)
            
            # Create metadata for the chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'paragraph_count': len(chunk_paragraphs),
                'start_paragraph': start,
                'end_paragraph': end,
            })
            
            # Create chunk
            chunks.append(TextChunk(
                id="",
                text=chunk_text,
                metadata=chunk_metadata
            ))
            
            # Update start position for next chunk, considering overlap
            start = end - self.overlap_paragraphs
            chunk_index += 1
        
        return chunks


class SemanticTextSplitter(ChunkingStrategy):
    """
    Split text based on semantic boundaries like headers, sections, or semantic similarity.
    This is a more advanced splitter that tries to preserve semantic coherence.
    """
    
    def __init__(self, 
                max_chunk_size: int = 1500,
                min_chunk_size: int = 200,
                header_patterns: Optional[List[str]] = None):
        """
        Initialize the text splitter.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters
            min_chunk_size: Minimum size of a chunk in characters
            header_patterns: List of regex patterns to identify headers/section breaks
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # Default header patterns (Markdown, HTML, etc.)
        if header_patterns is None:
            self.header_patterns = [
                r'^#{1,6}\s+.+$',  # Markdown headers
                r'^.+\n[=\-]{2,}$',  # Markdown underlining headers
                r'<h[1-6]>.*?</h[1-6]>',  # HTML headers
            ]
        else:
            self.header_patterns = header_patterns
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Split text into chunks based on semantic boundaries."""
        if metadata is None:
            metadata = {}
        
        # If text is shorter than max chunk size, return it as is
        if len(text) <= self.max_chunk_size:
            return [TextChunk(id="", text=text, metadata=metadata.copy())]
        
        # Find potential split points based on headers
        split_points = self._find_header_positions(text)
        
        # If no headers found, fall back to paragraph splitting
        if not split_points:
            paragraph_splitter = ParagraphTextSplitter()
            return paragraph_splitter.split_text(text, metadata)
        
        # Create chunks based on headers
        chunks = []
        start = 0
        chunk_index = 0
        
        # Add the text start as first split point if not already included
        if 0 not in split_points:
            split_points.insert(0, 0)
        
        # Add the text end as last split point if not already included
        if len(text) not in split_points:
            split_points.append(len(text))
        
        # Sort split points
        split_points.sort()
        
        # Create chunks from split points
        for i in range(len(split_points) - 1):
            section_start = split_points[i]
            section_end = split_points[i + 1]
            section_text = text[section_start:section_end].strip()
            
            # Skip empty sections
            if not section_text:
                continue
            
            # If section is very small, combine with the next section
            if len(section_text) < self.min_chunk_size and i < len(split_points) - 2:
                continue
            
            # If section is too large, split it further by paragraphs
            if len(section_text) > self.max_chunk_size:
                paragraph_splitter = ParagraphTextSplitter()
                section_chunks = paragraph_splitter.split_text(section_text, metadata)
                chunks.extend(section_chunks)
            else:
                # Create metadata for the chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                    'start_char': section_start,
                    'end_char': section_end,
                })
                
                # Create chunk
                chunks.append(TextChunk(
                    id="",
                    text=section_text,
                    metadata=chunk_metadata
                ))
            
            chunk_index += 1
        
        return chunks
    
    def _find_header_positions(self, text: str) -> List[int]:
        """
        Find positions of headers in the text.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of character positions where headers are found
        """
        positions = []
        
        # Check each line for header patterns
        lines = text.split('\n')
        char_index = 0
        
        for line in lines:
            # Check if line matches any header pattern
            for pattern in self.header_patterns:
                if re.match(pattern, line):
                    positions.append(char_index)
                    break
            
            # Move to next line
            char_index += len(line) + 1  # +1 for the newline
        
        return positions


class RecursiveTextSplitter(ChunkingStrategy):
    """
    Split text recursively using a sequence of separators.
    This approach tries different separators in order, falling back to simpler ones if needed.
    """
    
    def __init__(self, 
                chunk_size: int = 1000,
                chunk_overlap: int = 200,
                separators: Optional[List[str]] = None):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Maximum size of a chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try, in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, chunk_size // 2)
        
        # Default separators from most to least granular
        if separators is None:
            self.separators = [
                "\n\n",  # Paragraphs
                "\n",    # Lines
                ". ",    # Sentences
                ", ",    # Clauses
                " ",     # Words
                ""       # Characters
            ]
        else:
            self.separators = separators
    
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[TextChunk]:
        """Split text recursively using a sequence of separators."""
        if metadata is None:
            metadata = {}
        
        # If text is shorter than chunk size, return it as is
        if len(text) <= self.chunk_size:
            return [TextChunk(id="", text=text, metadata=metadata.copy())]
        
        # Try each separator in turn
        for separator in self.separators:
            if separator == "":
                # Last resort: character-level splitting
                return self._split_by_characters(text, metadata)
            
            # Split by current separator
            if separator in text:
                chunks = self._split_by_separator(text, separator, metadata)
                
                # Check if any chunk is still too large
                any_too_large = any(len(chunk.text) > self.chunk_size for chunk in chunks)
                
                if not any_too_large:
                    return chunks
                else:
                    # If some chunks are too large, recursively split them
                    # using the next separator
                    next_sep_idx = self.separators.index(separator) + 1
                    if next_sep_idx < len(self.separators):
                        next_separator = self.separators[next_sep_idx]
                        
                        # Create a splitter with the next separator
                        next_splitter = RecursiveTextSplitter(
                            chunk_size=self.chunk_size,
                            chunk_overlap=self.chunk_overlap,
                            separators=self.separators[next_sep_idx:]
                        )
                        
                        # Recursively split oversized chunks
                        result = []
                        for chunk in chunks:
                            if len(chunk.text) > self.chunk_size:
                                sub_chunks = next_splitter.split_text(chunk.text, chunk.metadata)
                                result.extend(sub_chunks)
                            else:
                                result.append(chunk)
                        
                        return result
        
        # Fallback to character splitting if no separator works
        return self._split_by_characters(text, metadata)
    
    def _split_by_separator(self, text: str, separator: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split text by a specific separator."""
        splits = text.split(separator)
        
        # If separator is empty, just wrap and return the text
        if separator == "":
            return [TextChunk(id="", text=text, metadata=metadata.copy())]
        
        # Recombine splits to respect chunk size
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for split in splits:
            # Calculate length if we add this split
            split_len = len(split)
            sep_len = len(separator)
            
            # Check if adding this split would exceed chunk size
            if current_length + split_len + (sep_len if current_chunk else 0) > self.chunk_size and current_chunk:
                # Join current chunk and add to result
                chunk_text = separator.join(current_chunk)
                
                # Create metadata for the chunk
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': chunk_index,
                })
                
                # Create chunk
                chunks.append(TextChunk(
                    id="",
                    text=chunk_text,
                    metadata=chunk_metadata
                ))
                
                # Start new chunk, considering overlap
                overlap_splits = []
                overlap_length = 0
                
                # Add the last few splits for overlap, as long as we don't exceed the overlap size
                for i in range(len(current_chunk) - 1, -1, -1):
                    if overlap_length + len(current_chunk[i]) + sep_len <= self.chunk_overlap:
                        overlap_splits.insert(0, current_chunk[i])
                        overlap_length += len(current_chunk[i]) + sep_len
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_splits
                current_length = overlap_length
                chunk_index += 1
            
            # Add split to current chunk
            if current_chunk:
                current_length += split_len + sep_len
            else:
                current_length += split_len
            
            current_chunk.append(split)
        
        # Add the last chunk if there's anything left
        if current_chunk:
            chunk_text = separator.join(current_chunk)
            
            # Create metadata for the chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
            })
            
            # Create chunk
            chunks.append(TextChunk(
                id="",
                text=chunk_text,
                metadata=chunk_metadata
            ))
        
        return chunks
    
    def _split_by_characters(self, text: str, metadata: Dict[str, Any]) -> List[TextChunk]:
        """Split text by characters as a last resort."""
        chunks = []
        chunk_index = 0
        
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            # Calculate chunk boundaries
            start = i
            end = min(i + self.chunk_size, len(text))
            
            # Extract chunk text
            chunk_text = text[start:end]
            
            # Create metadata for the chunk
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                'chunk_index': chunk_index,
                'start_char': start,
                'end_char': end,
            })
            
            # Create chunk
            chunks.append(TextChunk(
                id="",
                text=chunk_text,
                metadata=chunk_metadata
            ))
            
            chunk_index += 1
            
            # Stop if we've reached the end of the text
            if end == len(text):
                break
        
        return chunks


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""
    
    @staticmethod
    def create_strategy(strategy_name: str, **kwargs) -> ChunkingStrategy:
        """
        Create a chunking strategy based on name and parameters.
        
        Args:
            strategy_name: Name of the strategy to create
            **kwargs: Additional parameters for the strategy
            
        Returns:
            Configured chunking strategy
        """
        if strategy_name.lower() == "character":
            return CharacterTextSplitter(**kwargs)
        elif strategy_name.lower() == "token":
            return TokenTextSplitter(**kwargs)
        elif strategy_name.lower() == "sentence":
            return SentenceTextSplitter(**kwargs)
        elif strategy_name.lower() == "paragraph":
            return ParagraphTextSplitter(**kwargs)
        elif strategy_name.lower() == "semantic":
            return SemanticTextSplitter(**kwargs)
        elif strategy_name.lower() == "recursive":
            return RecursiveTextSplitter(**kwargs)
        else:
            # Default to recursive
            print(f"Strategy {strategy_name} not recognized, using recursive as default.")
            return RecursiveTextSplitter(**kwargs)


def compare_chunking_strategies(text: str) -> Dict[str, List[TextChunk]]:
    """
    Compare different chunking strategies on the same text.
    
    Args:
        text: Text to split
        
    Returns:
        Dictionary mapping strategy names to their resulting chunks
    """
    strategies = {
        "Character": CharacterTextSplitter(chunk_size=800, chunk_overlap=100),
        "Token": TokenTextSplitter(chunk_size=200, chunk_overlap=20),
        "Sentence": SentenceTextSplitter(max_sentences=10, overlap_sentences=2),
        "Paragraph": ParagraphTextSplitter(max_paragraphs=3, overlap_paragraphs=1),
        "Semantic": SemanticTextSplitter(max_chunk_size=800, min_chunk_size=200),
        "Recursive": RecursiveTextSplitter(chunk_size=800, chunk_overlap=100),
    }
    
    results = {}
    
    for name, strategy in strategies.items():
        print(f"Splitting with {name} strategy...")
        chunks = strategy.split_text(text, {"strategy": name})
        results[name] = chunks
        print(f"  Generated {len(chunks)} chunks")
    
    return results


if __name__ == "__main__":
    print("Chunking Strategies for RAG Systems")
    print("---------------------------------")
    
    # Example text for demonstration
    example_text = """
# Introduction to RAG Systems

Retrieval-Augmented Generation (RAG) combines retrieval-based methods with generative models to enhance the quality and factuality of generated content. By retrieving relevant information from external knowledge sources, RAG systems can provide more accurate and up-to-date responses.

## How RAG Works

The basic RAG process follows these steps:

1. Index a knowledge base by converting documents into vector embeddings
2. For each user query, retrieve relevant documents or passages
3. Augment the prompt to the language model with the retrieved context
4. Generate a response that's grounded in the retrieved information

This approach helps to overcome the limitations of pre-trained models by providing them with current, domain-specific information that might not have been part of their training data.

## Components of RAG Systems

### Document Processing
Effective document processing is crucial for RAG systems. This includes:
- Document loading from various sources
- Text extraction and cleaning
- Chunking strategies to break documents into manageable pieces
- Metadata enrichment to enhance retrieval

### Vector Databases
Vector databases store document embeddings and enable efficient similarity search:
- Different indexing approaches (HNSW, IVF, etc.)
- Scaling considerations for large collections
- Query optimization techniques

### Retrieval Methods
Retrieval approaches vary in complexity and effectiveness:
- Dense retrieval using vector similarity
- Sparse retrieval using keyword matching
- Hybrid approaches combining multiple methods
- Reranking to improve precision

### Augmentation Strategies
There are several ways to augment LLM prompts:
- Direct context insertion
- Few-shot examples based on retrieved content
- Tool augmentation for specialized tasks
- Knowledge graph integration

## Evaluating RAG Systems

Proper evaluation metrics include:
- Relevance of retrieved information
- Factual accuracy of generated responses
- Context utilization assessment
- Latency and performance considerations

## Conclusion

RAG represents an important advancement in LLM applications, bridging the gap between static pre-training and dynamic information needs. The approaches discussed in this document provide a foundation for building effective RAG systems across various domains.
    """
    
    # Compare strategies
    results = compare_chunking_strategies(example_text)
    
    # Show examples of different chunks
    print("\nComparing chunking results:\n")
    
    for strategy_name, chunks in results.items():
        print(f"{strategy_name} Strategy: {len(chunks)} chunks")
        if chunks:
            sample_chunk = chunks[0]
            print(f"  Sample chunk ({len(sample_chunk.text)} chars):")
            print(f"  {sample_chunk.text[:100]}...")
            print(f"  Metadata: {sample_chunk.metadata}")
        print("")
    
    # Demonstrate factory pattern
    print("\nUsing the ChunkingStrategyFactory:")
    strategy = ChunkingStrategyFactory.create_strategy(
        "semantic", 
        max_chunk_size=500, 
        min_chunk_size=100
    )
    factory_chunks = strategy.split_text(example_text, {"source": "factory_example"})
    print(f"Generated {len(factory_chunks)} chunks with factory-created semantic strategy")
    
    print("\nNote: The optimal chunking strategy depends on your specific use case.")
    print("Consider factors like document structure, retrieval requirements,")
    print("and the capabilities of your vector database when choosing a strategy.") 