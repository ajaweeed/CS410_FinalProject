"""
BM25 ranking implementation
"""

from typing import List, Dict, Tuple
import math
from indexing import DocumentIndex
from query_expansion import QueryExpander
from utils import tokenize
import config


class BM25Ranker:
    """
    BM25 ranking algorithm implementation.
    """
    
    def __init__(self, index: DocumentIndex, k1: float = None, b: float = None):
        """
        Initialize BM25 ranker.
        
        Args:
            index: DocumentIndex object
            k1: Term frequency saturation parameter
            b: Length normalization parameter
        """
        self.index = index
        self.k1 = k1 or config.BM25_K1
        self.b = b or config.BM25_B
        
    def calculate_bm25_score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate BM25 score for a document given query terms.
        
        Args:
            query_terms: List of query terms
            doc_id: Document identifier
            
        Returns:
            BM25 score
        """
        score = 0.0
        doc_length = self.index.doc_lengths[doc_id] if doc_id < len(self.index.doc_lengths) else 1
        avg_doc_length = self.index.avg_doc_length
        
        # Calculate length normalization factor
        length_norm = (1 - self.b) + self.b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
        
        # Calculate score for each query term
        for term in query_terms:
            if term not in self.index.vocabulary:
                continue
            
            # Term frequency in document
            tf = self.index.get_term_frequency(term, doc_id)
            
            if tf == 0:
                continue
            
            # Document frequency
            df = self.index.get_document_frequency(term)
            num_docs = self.index.get_num_documents()
            
            # IDF component: log((N - df + 0.5) / (df + 0.5))
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # TF component: (k1 + 1) * tf / (k1 * length_norm + tf)
            tf_component = ((self.k1 + 1) * tf) / (self.k1 * length_norm + tf)
            
            # BM25 score contribution
            score += idf * tf_component
        
        return score
    
    def rank_documents(self, query: str, query_expander: QueryExpander = None, 
                      expanded: bool = True, top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rank documents for a query using BM25.
        
        Args:
            query: Query string
            query_expander: QueryExpander instance (optional)
            expanded: Whether to use query expansion
            top_k: Number of top results to return (None for all)
            
        Returns:
            List of (doc_id, score) tuples sorted by score (descending)
        """
        # Get query terms
        if expanded and query_expander:
            query_terms = query_expander.expand_query(query, method="combined")
        else:
            query_terms = tokenize(query, remove_stopwords=True, lemmatize=True)
        
        if not query_terms:
            return []
        
        # Get candidate documents (documents containing at least one query term)
        candidate_doc_ids = set()
        for term in query_terms:
            if term in self.index.vocabulary:
                candidate_doc_ids.update(self.index.get_documents_with_term(term))
        
        # If no candidates found, return empty list
        if not candidate_doc_ids:
            return []
        
        # Calculate BM25 scores for all candidate documents
        scores = []
        for doc_id in candidate_doc_ids:
            score = self.calculate_bm25_score(query_terms, doc_id)
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        if top_k is not None:
            scores = scores[:top_k]
        
        return scores
    
    def rank_documents_with_expansion_weights(self, query: str, 
                                             query_expander: QueryExpander,
                                             expansion_scores: Dict[str, float],
                                             top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rank documents with weighted query expansion terms.
        
        Args:
            query: Query string
            query_expander: QueryExpander instance
            expansion_scores: Dictionary mapping expansion terms to their scores
            top_k: Number of top results to return (None for all)
            
        Returns:
            List of (doc_id, score) tuples sorted by score (descending)
        """
        # Get original query terms
        original_terms = tokenize(query, remove_stopwords=True, lemmatize=True)
        
        # Combine original terms and expansion terms with weights
        all_terms = set(original_terms)
        all_terms.update(expansion_scores.keys())
        
        # Get candidate documents
        candidate_doc_ids = set()
        for term in all_terms:
            if term in self.index.vocabulary:
                candidate_doc_ids.update(self.index.get_documents_with_term(term))
        
        if not candidate_doc_ids:
            return []
        
        # Calculate weighted BM25 scores
        scores = []
        for doc_id in candidate_doc_ids:
            score = 0.0
            doc_length = self.index.doc_lengths[doc_id] if doc_id < len(self.index.doc_lengths) else 1
            avg_doc_length = self.index.avg_doc_length
            length_norm = (1 - self.b) + self.b * (doc_length / avg_doc_length) if avg_doc_length > 0 else 1.0
            
            # Score original terms with weight 1.0
            for term in original_terms:
                if term not in self.index.vocabulary:
                    continue
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue
                df = self.index.get_document_frequency(term)
                num_docs = self.index.get_num_documents()
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf_component = ((self.k1 + 1) * tf) / (self.k1 * length_norm + tf)
                score += idf * tf_component
            
            # Score expansion terms with their weights
            for term, term_weight in expansion_scores.items():
                if term not in self.index.vocabulary:
                    continue
                tf = self.index.get_term_frequency(term, doc_id)
                if tf == 0:
                    continue
                df = self.index.get_document_frequency(term)
                num_docs = self.index.get_num_documents()
                idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                tf_component = ((self.k1 + 1) * tf) / (self.k1 * length_norm + tf)
                score += term_weight * idf * tf_component
            
            if score > 0:
                scores.append((doc_id, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        if top_k is not None:
            scores = scores[:top_k]
        
        return scores




