"""
Query expansion module using WordNet and GloVe embeddings
"""

import os
import numpy as np
from typing import List, Dict, Set, Tuple
from nltk.corpus import wordnet as wn
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import requests
import zipfile
import config
from utils import tokenize, ensure_dir


class QueryExpander:
    """
    Query expansion using WordNet synonyms and GloVe embeddings.
    """
    
    def __init__(self, embedding_dim: int = None):
        """
        Initialize the query expander.
        
        Args:
            embedding_dim: Dimension of GloVe embeddings (50, 100, 200, or 300)
        """
        self.embedding_dim = embedding_dim or config.GLOVE_DIM
        self.word_vectors = None
        self._load_embeddings()
    
    def _load_embeddings(self):
        """
        Load GloVe embeddings, downloading if necessary.
        """
        embeddings_dir = config.EMBEDDINGS_CACHE_DIR
        ensure_dir(embeddings_dir)
        
        # Check if embeddings are already loaded
        glove_file = os.path.join(embeddings_dir, f"glove.6B.{self.embedding_dim}d.txt")
        w2v_file = os.path.join(embeddings_dir, f"glove.6B.{self.embedding_dim}d.w2v.txt")
        
        # Convert GloVe to Word2Vec format if needed
        if os.path.exists(glove_file) and not os.path.exists(w2v_file):
            print(f"Converting GloVe format to Word2Vec format...")
            glove2word2vec(glove_file, w2v_file)
        
        # Load embeddings
        if os.path.exists(w2v_file):
            print(f"Loading GloVe embeddings ({self.embedding_dim}d)...")
            self.word_vectors = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
            print(f"Loaded {len(self.word_vectors)} word vectors")
        else:
            print(f"Warning: GloVe embeddings not found at {w2v_file}")
            print("Embedding-based expansion will be disabled.")
            print("To enable, download GloVe embeddings from:")
            print("https://nlp.stanford.edu/projects/glove/")
            print(f"Extract glove.6B.{self.embedding_dim}d.txt to {embeddings_dir}/")
    
    def expand_with_wordnet(self, query_terms: List[str], top_k: int = None) -> List[str]:
        """
        Expand query using WordNet synonyms.
        
        Args:
            query_terms: List of query terms
            top_k: Maximum number of expansion terms to return per query term
            
        Returns:
            List of expanded terms (including original terms)
        """
        if top_k is None:
            top_k = config.EXPANSION_TOP_K
        
        expanded_terms = set(query_terms)  # Start with original terms
        
        for term in query_terms:
            # Get synonyms from WordNet
            synonyms = set()
            for syn in wn.synsets(term):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ').lower()
                    # Filter out the original term and single characters
                    if synonym != term and len(synonym) >= 2:
                        synonyms.add(synonym)
            
            # Add top synonyms
            expanded_terms.update(list(synonyms)[:top_k])
        
        return list(expanded_terms)
    
    def expand_with_embeddings(self, query_terms: List[str], top_k: int = None, 
                               similarity_threshold: float = None) -> List[Tuple[str, float]]:
        """
        Expand query using GloVe embeddings.
        
        Args:
            query_terms: List of query terms
            top_k: Maximum number of expansion terms to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (term, similarity_score) tuples
        """
        if self.word_vectors is None:
            return []
        
        if top_k is None:
            top_k = config.EXPANSION_TOP_K
        
        if similarity_threshold is None:
            similarity_threshold = config.EMBEDDING_SIMILARITY_THRESHOLD
        
        # Filter query terms that exist in the vocabulary
        valid_terms = [term for term in query_terms if term in self.word_vectors]
        
        if not valid_terms:
            return []
        
        # Get average vector of query terms
        query_vector = np.mean([self.word_vectors[term] for term in valid_terms], axis=0)
        
        # Find most similar words
        try:
            similar_words = self.word_vectors.similar_by_vector(
                query_vector, 
                topn=top_k * 2  # Get more to filter
            )
            
            # Filter by threshold and exclude original terms
            expanded_terms = [
                (word, score) for word, score in similar_words
                if score >= similarity_threshold and word not in query_terms
            ][:top_k]
            
            return expanded_terms
        except Exception as e:
            print(f"Error in embedding expansion: {e}")
            return []
    
    def expand_query(self, query: str, method: str = "combined", 
                    wordnet_weight: float = None, embedding_weight: float = None) -> List[str]:
        """
        Expand a query using the specified method.
        
        Args:
            query: Query string
            method: Expansion method ("wordnet", "embedding", or "combined")
            wordnet_weight: Weight for WordNet expansion (for combined method)
            embedding_weight: Weight for embedding expansion (for combined method)
            
        Returns:
            List of expanded query terms
        """
        if wordnet_weight is None:
            wordnet_weight = config.WORDNET_WEIGHT
        if embedding_weight is None:
            embedding_weight = config.EMBEDDING_WEIGHT
        
        # Tokenize query
        query_terms = tokenize(query, remove_stopwords=False, lemmatize=True)
        
        if not query_terms:
            return []
        
        expanded_terms = set(query_terms)  # Always include original terms
        
        if method == "wordnet" or method == "combined":
            # WordNet expansion
            wordnet_terms = self.expand_with_wordnet(query_terms)
            expanded_terms.update(wordnet_terms)
        
        if method == "embedding" or method == "combined":
            # Embedding expansion
            embedding_results = self.expand_with_embeddings(query_terms)
            embedding_terms = [term for term, score in embedding_results]
            expanded_terms.update(embedding_terms)
        
        return list(expanded_terms)
    
    def get_expansion_scores(self, query: str) -> Dict[str, float]:
        """
        Get expansion terms with their scores.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary mapping expansion terms to their scores
        """
        query_terms = tokenize(query, remove_stopwords=False, lemmatize=True)
        
        expansion_scores = {}
        
        # WordNet expansion (all terms get equal weight)
        wordnet_terms = self.expand_with_wordnet(query_terms)
        for term in wordnet_terms:
            if term not in query_terms:
                expansion_scores[term] = expansion_scores.get(term, 0) + config.WORDNET_WEIGHT
        
        # Embedding expansion (weighted by similarity)
        embedding_results = self.expand_with_embeddings(query_terms)
        for term, score in embedding_results:
            expansion_scores[term] = expansion_scores.get(term, 0) + config.EMBEDDING_WEIGHT * score
        
        return expansion_scores




