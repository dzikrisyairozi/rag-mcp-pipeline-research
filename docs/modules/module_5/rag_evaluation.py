"""
RAG Evaluation Framework

This module provides tools and metrics for evaluating Retrieval Augmented Generation (RAG)
systems, including relevance metrics, hallucination detection, and context utilization assessment.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable
import re
import numpy as np
import time


@dataclass
class Document:
    """Represents a document in the retrieval system."""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""
    query: str
    documents: List[Document]
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass
class GenerationResult:
    """Result of a generation operation."""
    query: str
    context: str
    response: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time_ms: float = 0.0


@dataclass
class EvaluationScore:
    """Score for a single evaluation metric."""
    name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """Result of evaluating a RAG system on a query."""
    query: str
    retrieval_result: RetrievalResult
    generation_result: GenerationResult
    scores: List[EvaluationScore] = field(default_factory=list)
    
    def add_score(self, name: str, score: float, details: Dict[str, Any] = None):
        """Add a score to the evaluation result."""
        self.scores.append(EvaluationScore(
            name=name,
            score=score,
            details=details or {}
        ))
    
    def get_score(self, name: str) -> Optional[EvaluationScore]:
        """Get a score by name."""
        for score in self.scores:
            if score.name == name:
                return score
        return None
    
    def get_average_score(self) -> float:
        """Get the average of all scores."""
        if not self.scores:
            return 0.0
        return sum(score.score for score in self.scores) / len(self.scores)


class EvaluationMetric(ABC):
    """Abstract base class for RAG evaluation metrics."""
    
    @abstractmethod
    def evaluate(self, result: EvaluationResult) -> float:
        """
        Evaluate a RAG result and return a score.
        
        Args:
            result: The evaluation result to score
            
        Returns:
            Score between 0.0 and 1.0
        """
        pass


class RelevanceMetric(EvaluationMetric):
    """Evaluates the relevance of retrieved documents to the query."""
    
    def __init__(self, relevance_fn: Optional[Callable[[str, str], float]] = None):
        """
        Initialize the relevance metric.
        
        Args:
            relevance_fn: Optional function to calculate relevance
        """
        self.relevance_fn = relevance_fn
    
    def evaluate(self, result: EvaluationResult) -> float:
        """Evaluate the relevance of retrieved documents."""
        query = result.query
        documents = result.retrieval_result.documents
        
        if not documents:
            return 0.0
        
        # Calculate relevance scores for each document
        relevance_scores = []
        
        for doc in documents:
            if self.relevance_fn:
                # Use provided relevance function
                score = self.relevance_fn(query, doc.content)
            else:
                # Use simple keyword matching
                score = self._keyword_relevance(query, doc.content)
            
            relevance_scores.append(score)
        
        # Calculate average relevance
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Add details to the evaluation result
        result.add_score("relevance", avg_relevance, {
            "document_scores": {doc.id: score for doc, score in zip(documents, relevance_scores)},
            "query": query
        })
        
        return avg_relevance
    
    def _keyword_relevance(self, query: str, content: str) -> float:
        """
        Calculate relevance based on keyword matching.
        
        Args:
            query: The search query
            content: Document content
            
        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Extract query terms (excluding common stop words)
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        
        # Remove very common words
        stop_words = {'the', 'a', 'an', 'in', 'to', 'for', 'of', 'and', 'is', 'are'}
        query_terms = query_terms - stop_words
        
        if not query_terms:
            return 0.5  # Neutral score if no meaningful terms
        
        # Count matching terms
        matches = 0
        for term in query_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', content_lower):
                matches += 1
        
        # Calculate relevance score
        return matches / len(query_terms)


class HallucinationDetector(EvaluationMetric):
    """Detects potential hallucinations in generated responses."""
    
    def __init__(self, strictness: float = 0.7):
        """
        Initialize the hallucination detector.
        
        Args:
            strictness: How strict to be in hallucination detection (0.0-1.0)
        """
        self.strictness = strictness
    
    def evaluate(self, result: EvaluationResult) -> float:
        """
        Evaluate potential hallucinations in the response.
        
        Returns a score where 1.0 means no hallucinations and 0.0 means severe hallucinations.
        """
        response = result.generation_result.response
        context = result.generation_result.context
        
        # Extract factual claims from the response
        claims = self._extract_claims(response)
        
        if not claims:
            return 1.0  # No claims, so no hallucinations
        
        # Check each claim against the context
        supported_claims = 0
        claim_scores = {}
        
        for claim in claims:
            support_score = self._check_claim_support(claim, context)
            claim_scores[claim] = support_score
            
            if support_score >= self.strictness:
                supported_claims += 1
        
        # Calculate hallucination score (inverse of supported ratio)
        hallucination_score = supported_claims / len(claims)
        
        # Add details to the evaluation result
        result.add_score("factual_consistency", hallucination_score, {
            "claims": claim_scores,
            "strictness": self.strictness
        })
        
        return hallucination_score
    
    def _extract_claims(self, text: str) -> List[str]:
        """
        Extract factual claims from text.
        
        This is a simplified implementation that treats sentences with certain patterns
        as claims. A more sophisticated approach would use NLP techniques.
        
        Args:
            text: Text to extract claims from
            
        Returns:
            List of claim statements
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter for sentences that appear to be factual claims
        claims = []
        
        claim_indicators = [
            r'is a', r'are a', r'was a', r'were a',
            r'contains', r'consists of', r'comprises',
            r'has', r'have', r'had',
            r'provides', r'offer', r'support',
            r'shows', r'demonstrates', r'proves',
            r'according to', r'stated in', r'mentioned in'
        ]
        
        pattern = '|'.join(claim_indicators)
        
        for sentence in sentences:
            # Filter out questions and very short sentences
            if sentence.endswith('?') or len(sentence.split()) < 4:
                continue
                
            # Check if it contains claim indicators
            if re.search(pattern, sentence.lower()):
                claims.append(sentence)
        
        return claims
    
    def _check_claim_support(self, claim: str, context: str) -> float:
        """
        Check if a claim is supported by the context.
        
        Args:
            claim: The claim to check
            context: The context to check against
            
        Returns:
            Support score between 0.0 and 1.0
        """
        # Convert to lowercase for case-insensitive matching
        claim_lower = claim.lower()
        context_lower = context.lower()
        
        # Extract key terms from the claim
        claim_terms = set(re.findall(r'\b\w+\b', claim_lower))
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'in', 'to', 'for', 'of', 'and', 'is', 'are'}
        claim_terms = claim_terms - stop_words
        
        if not claim_terms:
            return 0.5  # Neutral score if no meaningful terms
        
        # Count matching terms
        matches = 0
        for term in claim_terms:
            if re.search(r'\b' + re.escape(term) + r'\b', context_lower):
                matches += 1
        
        # Calculate support score
        return matches / len(claim_terms)


class ContextUtilizationMetric(EvaluationMetric):
    """Evaluates how effectively the generation utilizes the provided context."""
    
    def evaluate(self, result: EvaluationResult) -> float:
        """
        Evaluate context utilization in the response.
        
        Returns a score where 1.0 means perfect utilization and 0.0 means no utilization.
        """
        response = result.generation_result.response
        context = result.generation_result.context
        
        # Extract key information from context
        context_info = self._extract_key_info(context)
        
        if not context_info:
            return 0.5  # Neutral score if no key info
        
        # Check how much of the key info is used in the response
        response_lower = response.lower()
        
        used_info = 0
        info_scores = {}
        
        for info in context_info:
            # Check if the info or its synonyms are in the response
            if self._check_info_usage(info, response_lower):
                used_info += 1
                info_scores[info] = 1.0
            else:
                info_scores[info] = 0.0
        
        # Calculate utilization score
        utilization_score = used_info / len(context_info)
        
        # Add details to the evaluation result
        result.add_score("context_utilization", utilization_score, {
            "info_usage": info_scores
        })
        
        return utilization_score
    
    def _extract_key_info(self, context: str) -> List[str]:
        """
        Extract key information pieces from the context.
        
        This is a simplified implementation that extracts entities and key phrases.
        A more sophisticated approach would use NLP techniques for entity extraction.
        
        Args:
            context: Context text to analyze
            
        Returns:
            List of key information pieces
        """
        # Convert to lowercase
        context_lower = context.lower()
        
        # Extract potential entities (capitalized words not at start of sentence)
        entities = re.findall(r'(?<!\. )[A-Z][a-z]+', context)
        
        # Extract phrases with keywords
        key_phrases = []
        
        info_indicators = [
            r'is a', r'are a', r'was a', r'were a',
            r'contains', r'consists of', r'comprises',
            r'has', r'have', r'had',
            r'provides', r'offer', r'support',
            r'shows', r'demonstrates', r'proves'
        ]
        
        pattern = '|'.join(info_indicators)
        
        # Find phrases containing these patterns
        sentences = re.split(r'(?<=[.!?])\s+', context)
        for sentence in sentences:
            if re.search(pattern, sentence.lower()):
                # Extract subject-verb-object structures (simplified)
                matches = re.findall(r'([A-Za-z ]+)(' + pattern + r')([A-Za-z ]+)', sentence.lower())
                for match in matches:
                    key_phrases.append(''.join(match).strip())
        
        # Combine entities and key phrases, removing duplicates
        key_info = list(set(entities + key_phrases))
        
        # Filter out very short or common items
        return [info for info in key_info if len(info) > 3]
    
    def _check_info_usage(self, info: str, response: str) -> bool:
        """
        Check if information is used in the response.
        
        Args:
            info: Information to check for
            response: Response to check in
            
        Returns:
            True if the information is used, False otherwise
        """
        info_lower = info.lower()
        
        # Direct match
        if info_lower in response:
            return True
        
        # Check for partial matches (for longer phrases)
        if len(info_lower) > 10:
            words = info_lower.split()
            if len(words) > 3:
                # Check if most words are present
                matches = 0
                for word in words:
                    if len(word) > 3 and re.search(r'\b' + re.escape(word) + r'\b', response):
                        matches += 1
                
                if matches >= len(words) * 0.7:
                    return True
        
        return False


class ResponseQualityMetric(EvaluationMetric):
    """Evaluates the overall quality of the generated response."""
    
    def __init__(self, fluency_weight: float = 0.3, informativeness_weight: float = 0.7):
        """
        Initialize the response quality metric.
        
        Args:
            fluency_weight: Weight for fluency in the quality score
            informativeness_weight: Weight for informativeness in the quality score
        """
        self.fluency_weight = fluency_weight
        self.informativeness_weight = informativeness_weight
    
    def evaluate(self, result: EvaluationResult) -> float:
        """
        Evaluate the quality of the response.
        
        Returns a score where 1.0 means excellent quality and 0.0 means poor quality.
        """
        response = result.generation_result.response
        query = result.query
        
        # Calculate fluency score
        fluency_score = self._evaluate_fluency(response)
        
        # Calculate informativeness score
        informativeness_score = self._evaluate_informativeness(response, query)
        
        # Calculate overall quality score
        quality_score = (
            self.fluency_weight * fluency_score +
            self.informativeness_weight * informativeness_score
        )
        
        # Add details to the evaluation result
        result.add_score("response_quality", quality_score, {
            "fluency": fluency_score,
            "informativeness": informativeness_score
        })
        
        return quality_score
    
    def _evaluate_fluency(self, text: str) -> float:
        """
        Evaluate the fluency of text.
        
        This is a simplified implementation that checks for basic indicators of fluency.
        A more sophisticated approach would use language models or grammatical analysis.
        
        Args:
            text: Text to evaluate
            
        Returns:
            Fluency score between 0.0 and 1.0
        """
        if not text:
            return 0.0
        
        # Check for very short sentences (potential fragments)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        very_short = sum(1 for s in sentences if len(s.split()) < 3)
        
        # Calculate ratio of very short sentences
        short_ratio = very_short / len(sentences) if sentences else 0
        
        # Check for repeated words (potential disfluency)
        words = re.findall(r'\b\w+\b', text.lower())
        repeated_count = 0
        
        for i in range(len(words) - 1):
            if words[i] == words[i + 1]:
                repeated_count += 1
        
        repeated_ratio = repeated_count / len(words) if words else 0
        
        # Check average sentence length (very long or very short sentences can indicate disfluency)
        avg_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        length_score = 0.0
        
        if avg_length > 5 and avg_length < 25:
            length_score = 1.0
        elif avg_length > 3 and avg_length < 30:
            length_score = 0.7
        else:
            length_score = 0.3
        
        # Calculate overall fluency score
        fluency_score = (1.0 - short_ratio) * 0.3 + (1.0 - repeated_ratio) * 0.3 + length_score * 0.4
        
        return min(max(fluency_score, 0.0), 1.0)
    
    def _evaluate_informativeness(self, response: str, query: str) -> float:
        """
        Evaluate how informative the response is.
        
        This is a simplified implementation that checks if the response seems to address the query.
        A more sophisticated approach would compare to reference answers or use semantic analysis.
        
        Args:
            response: Response to evaluate
            query: Query that should be addressed
            
        Returns:
            Informativeness score between 0.0 and 1.0
        """
        if not response:
            return 0.0
        
        # Count sentences (more sentences suggest more information)
        sentences = re.split(r'(?<=[.!?])\s+', response)
        sentence_count = len(sentences)
        
        # Short responses may not be informative
        if sentence_count < 2:
            sentence_score = 0.5
        elif sentence_count < 4:
            sentence_score = 0.8
        else:
            sentence_score = 1.0
        
        # Check if response contains query terms (addressing the query)
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        response_lower = response.lower()
        
        # Remove very common words
        stop_words = {'the', 'a', 'an', 'in', 'to', 'for', 'of', 'and', 'is', 'are', 'what', 'why', 'how', 'when', 'where'}
        query_terms = query_terms - stop_words
        
        if not query_terms:
            query_term_score = 0.5  # Neutral score if no meaningful terms
        else:
            # Count matching terms
            matches = 0
            for term in query_terms:
                if re.search(r'\b' + re.escape(term) + r'\b', response_lower):
                    matches += 1
            
            query_term_score = matches / len(query_terms)
        
        # Calculate informativeness score
        informativeness_score = sentence_score * 0.4 + query_term_score * 0.6
        
        return min(max(informativeness_score, 0.0), 1.0)


class RAGEvaluator:
    """Main class for evaluating RAG systems."""
    
    def __init__(self, metrics: Optional[List[EvaluationMetric]] = None):
        """
        Initialize the RAG evaluator.
        
        Args:
            metrics: List of evaluation metrics to use
        """
        self.metrics = metrics or [
            RelevanceMetric(),
            HallucinationDetector(),
            ContextUtilizationMetric(),
            ResponseQualityMetric()
        ]
    
    def evaluate(self, query: str, retrieval_result: RetrievalResult, 
                generation_result: GenerationResult) -> EvaluationResult:
        """
        Evaluate a RAG system on a single query.
        
        Args:
            query: The query
            retrieval_result: Result of the retrieval step
            generation_result: Result of the generation step
            
        Returns:
            Evaluation result with scores from all metrics
        """
        # Create evaluation result
        result = EvaluationResult(
            query=query,
            retrieval_result=retrieval_result,
            generation_result=generation_result
        )
        
        # Apply each metric
        for metric in self.metrics:
            metric.evaluate(result)
        
        return result
    
    def evaluate_batch(self, queries: List[str], 
                      retrieval_results: List[RetrievalResult],
                      generation_results: List[GenerationResult]) -> List[EvaluationResult]:
        """
        Evaluate a RAG system on multiple queries.
        
        Args:
            queries: List of queries
            retrieval_results: List of retrieval results
            generation_results: List of generation results
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for query, retrieval, generation in zip(queries, retrieval_results, generation_results):
            results.append(self.evaluate(query, retrieval, generation))
        
        return results
    
    def summarize_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Summarize evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        # Collect scores by metric
        scores_by_metric = {}
        
        for result in results:
            for score in result.scores:
                if score.name not in scores_by_metric:
                    scores_by_metric[score.name] = []
                scores_by_metric[score.name].append(score.score)
        
        # Calculate summary statistics
        summary = {}
        
        for metric_name, scores in scores_by_metric.items():
            summary[metric_name] = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
                "std": np.std(scores) if len(scores) > 1 else 0.0
            }
        
        # Add overall score
        all_scores = [score for result in results for score in result.scores]
        summary["overall"] = {
            "mean": sum(score.score for score in all_scores) / len(all_scores) if all_scores else 0.0
        }
        
        return summary


def create_sample_retrieval_result(query: str, num_documents: int = 3) -> RetrievalResult:
    """Create a sample retrieval result for testing."""
    documents = []
    
    for i in range(num_documents):
        doc = Document(
            id=f"doc_{i+1}",
            content=f"This is document {i+1} about {query}. It contains information relevant to the query.",
            score=0.9 - (i * 0.1)
        )
        documents.append(doc)
    
    return RetrievalResult(
        query=query,
        documents=documents,
        latency_ms=100.0
    )


def create_sample_generation_result(query: str, retrieval_result: RetrievalResult) -> GenerationResult:
    """Create a sample generation result for testing."""
    # Combine document contents for context
    context = "\n\n".join([doc.content for doc in retrieval_result.documents])
    
    # Generate a simple response
    response = f"Based on the provided information, {query} refers to a concept in AI that has several applications."
    
    return GenerationResult(
        query=query,
        context=context,
        response=response,
        generation_time_ms=200.0
    )


def demonstrate_rag_evaluation():
    """Demonstrate the RAG evaluation framework."""
    print("RAG Evaluation Framework Demonstration")
    print("--------------------------------------")
    
    # Create sample data
    queries = [
        "What is retrieval augmented generation?",
        "How do vector databases work?",
        "What are embedding models?"
    ]
    
    retrieval_results = [create_sample_retrieval_result(query) for query in queries]
    generation_results = [create_sample_generation_result(query, result) for query, result in zip(queries, retrieval_results)]
    
    # Create evaluator
    evaluator = RAGEvaluator()
    
    # Evaluate
    evaluation_results = evaluator.evaluate_batch(queries, retrieval_results, generation_results)
    
    # Print results
    for i, result in enumerate(evaluation_results):
        print(f"\nQuery {i+1}: {result.query}")
        print("-" * 40)
        for score in result.scores:
            print(f"{score.name}: {score.score:.4f}")
    
    # Print summary
    summary = evaluator.summarize_results(evaluation_results)
    
    print("\nSummary Statistics:")
    print("-" * 40)
    for metric_name, stats in summary.items():
        if metric_name != "overall":
            print(f"{metric_name}: mean={stats['mean']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")
    
    print(f"\nOverall score: {summary['overall']['mean']:.4f}")


if __name__ == "__main__":
    demonstrate_rag_evaluation() 