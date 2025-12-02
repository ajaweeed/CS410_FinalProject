"""
Document indexing module for the search engine
"""

from typing import List, Dict, Set
from collections import defaultdict
import math
from utils import preprocess_document, calculate_term_frequency
import config


class DocumentIndex:
    """
    Index for storing and retrieving document information.
    """
    
    def __init__(self):
        self.documents: List[str] = []
        self.processed_docs: List[List[str]] = []
        self.vocabulary: Set[str] = set()
        self.term_doc_freq: Dict[str, int] = defaultdict(int)  # Document frequency
        self.term_freq: List[Dict[str, int]] = []  # Term frequency per document
        self.doc_lengths: List[int] = []
        self.avg_doc_length: float = 0.0
        self.inverse_index: Dict[str, List[int]] = defaultdict(list)  # Term -> doc_ids
        self.tf_idf_matrix: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.idf: Dict[str, float] = {}
        
    def add_document(self, doc_id: int, document: str):
        """
        Add a document to the index.
        
        Args:
            doc_id: Document identifier
            document: Document text
        """
        # Preprocess document
        tokens = preprocess_document(document)
        
        # Store document and processed tokens
        if doc_id >= len(self.documents):
            self.documents.extend([''] * (doc_id - len(self.documents) + 1))
            self.processed_docs.extend([[]] * (doc_id - len(self.processed_docs) + 1))
            self.term_freq.extend([{}] * (doc_id - len(self.term_freq) + 1))
            self.doc_lengths.extend([0] * (doc_id - len(self.doc_lengths) + 1))
        
        self.documents[doc_id] = document
        self.processed_docs[doc_id] = tokens
        
        # Calculate term frequencies
        tf = calculate_term_frequency(tokens)
        self.term_freq[doc_id] = tf
        
        # Update vocabulary and document frequency
        doc_terms = set(tf.keys())
        for term in doc_terms:
            self.vocabulary.add(term)
            self.term_doc_freq[term] += 1
            self.inverse_index[term].append(doc_id)
        
        # Store document length
        self.doc_lengths[doc_id] = len(tokens)
    
    def build_index(self, documents: List[str]):
        """
        Build the index from a list of documents.
        
        Args:
            documents: List of document strings
        """
        self.documents = documents
        num_docs = len(documents)
        
        # Process all documents
        for doc_id, document in enumerate(documents):
            tokens = preprocess_document(document)
            self.processed_docs.append(tokens)
            
            # Calculate term frequencies
            tf = calculate_term_frequency(tokens)
            self.term_freq.append(tf)
            
            # Update vocabulary and document frequency
            doc_terms = set(tf.keys())
            for term in doc_terms:
                self.vocabulary.add(term)
                self.term_doc_freq[term] += 1
                self.inverse_index[term].append(doc_id)
            
            # Store document length
            doc_length = len(tokens)
            self.doc_lengths.append(doc_length)
        
        # Filter vocabulary by document frequency
        self._filter_vocabulary(num_docs)
        
        # Calculate average document length
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        
        # Calculate IDF and TF-IDF
        self._calculate_idf(num_docs)
        self._calculate_tf_idf(num_docs)
    
    def _filter_vocabulary(self, num_docs: int):
        """
        Filter vocabulary based on document frequency thresholds.
        
        Args:
            num_docs: Total number of documents
        """
        min_df_count = max(1, int(config.MIN_DF))
        max_df_count = int(config.MAX_DF * num_docs)
        
        filtered_vocab = set()
        filtered_term_doc_freq = defaultdict(int)
        filtered_inverse_index = defaultdict(list)
        
        for term in self.vocabulary:
            df = self.term_doc_freq[term]
            if min_df_count <= df <= max_df_count:
                filtered_vocab.add(term)
                filtered_term_doc_freq[term] = df
                filtered_inverse_index[term] = self.inverse_index[term]
        
        self.vocabulary = filtered_vocab
        self.term_doc_freq = filtered_term_doc_freq
        self.inverse_index = filtered_inverse_index
    
    def _calculate_idf(self, num_docs: int):
        """
        Calculate Inverse Document Frequency (IDF) for all terms.
        
        Args:
            num_docs: Total number of documents
        """
        for term in self.vocabulary:
            df = self.term_doc_freq[term]
            # IDF = log((N + 1) / (df + 1))
            self.idf[term] = math.log((num_docs + 1) / (df + 1))
    
    def _calculate_tf_idf(self, num_docs: int):
        """
        Calculate TF-IDF weights for all terms in all documents.
        
        Args:
            num_docs: Total number of documents
        """
        for doc_id, tf_dict in enumerate(self.term_freq):
            doc_length = self.doc_lengths[doc_id] if doc_id < len(self.doc_lengths) else 0
            
            for term, tf_count in tf_dict.items():
                if term in self.vocabulary:
                    # Normalized TF
                    normalized_tf = tf_count / doc_length if doc_length > 0 else 0
                    
                    # TF-IDF = TF * IDF
                    self.tf_idf_matrix[term][doc_id] = normalized_tf * self.idf[term]
    
    def get_document_vector(self, doc_id: int) -> Dict[str, float]:
        """
        Get the TF-IDF vector for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Dictionary mapping terms to TF-IDF weights
        """
        vector = {}
        if doc_id < len(self.term_freq):
            for term, tf_count in self.term_freq[doc_id].items():
                if term in self.vocabulary:
                    vector[term] = self.tf_idf_matrix[term].get(doc_id, 0.0)
        return vector
    
    def get_documents_with_term(self, term: str) -> List[int]:
        """
        Get document IDs that contain a specific term.
        
        Args:
            term: Term to search for
            
        Returns:
            List of document IDs
        """
        return self.inverse_index.get(term, [])
    
    def get_term_frequency(self, term: str, doc_id: int) -> int:
        """
        Get term frequency in a specific document.
        
        Args:
            term: Term to look up
            doc_id: Document identifier
            
        Returns:
            Term frequency in the document
        """
        if doc_id < len(self.term_freq):
            return self.term_freq[doc_id].get(term, 0)
        return 0
    
    def get_document_frequency(self, term: str) -> int:
        """
        Get document frequency of a term.
        
        Args:
            term: Term to look up
            
        Returns:
            Document frequency
        """
        return self.term_doc_freq.get(term, 0)
    
    def get_idf(self, term: str) -> float:
        """
        Get IDF value for a term.
        
        Args:
            term: Term to look up
            
        Returns:
            IDF value
        """
        return self.idf.get(term, 0.0)
    
    def get_num_documents(self) -> int:
        """
        Get the total number of documents in the index.
        
        Returns:
            Number of documents
        """
        return len(self.documents)
    
    def get_vocabulary_size(self) -> int:
        """
        Get the size of the vocabulary.
        
        Returns:
            Vocabulary size
        """
        return len(self.vocabulary)




