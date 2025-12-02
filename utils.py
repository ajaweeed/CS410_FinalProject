"""
Utility functions for the search engine
"""

import re
import os
import pickle
from typing import List, Dict, Set
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download required NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Get stopwords
STOPWORDS = set(stopwords.words('english'))


def load_documents(file_path: str) -> List[str]:
    """
    Load documents from a text file.
    
    Args:
        file_path: Path to the file containing documents (one per line)
        
    Returns:
        List of document strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]
    
    return documents


def tokenize(text: str, remove_stopwords: bool = True, lemmatize: bool = True) -> List[str]:
    """
    Tokenize and preprocess text.
    
    Args:
        text: Input text string
        remove_stopwords: Whether to remove stopwords
        lemmatize: Whether to lemmatize tokens
        
    Returns:
        List of processed tokens
    """
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove non-alphanumeric tokens and filter by length
    tokens = [token for token in tokens if token.isalnum() and len(token) >= 2]
    
    # Remove stopwords if requested
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    
    # Lemmatize if requested
    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return tokens


def preprocess_document(doc: str) -> List[str]:
    """
    Preprocess a single document.
    
    Args:
        doc: Document text
        
    Returns:
        List of processed tokens
    """
    return tokenize(doc, remove_stopwords=True, lemmatize=True)


def save_object(obj: object, file_path: str):
    """
    Save an object to disk using pickle.
    
    Args:
        obj: Object to save
        file_path: Path to save the object
    """
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)


def load_object(file_path: str) -> object:
    """
    Load an object from disk using pickle.
    
    Args:
        file_path: Path to load the object from
        
    Returns:
        Loaded object
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Object file not found: {file_path}")
    
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def ensure_dir(directory: str):
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def calculate_term_frequency(tokens: List[str]) -> Dict[str, int]:
    """
    Calculate term frequency for a list of tokens.
    
    Args:
        tokens: List of tokens
        
    Returns:
        Dictionary mapping terms to their frequencies
    """
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    return tf


def load_queries(file_path: str) -> List[str]:
    """
    Load queries from a text file.
    
    Args:
        file_path: Path to the file containing queries (one per line)
        
    Returns:
        List of query strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Query file not found: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    return queries


def load_relevance_judgments(file_path: str) -> Dict[int, Dict[int, int]]:
    """
    Load relevance judgments from a file.
    Format: query_id\tdoc_id\trelevance_score
    
    Args:
        file_path: Path to the relevance judgments file
        
    Returns:
        Dictionary mapping query_id to doc_id to relevance_score
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Relevance file not found: {file_path}")
    
    relevance = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id = int(parts[0])
                doc_id = int(parts[1])
                score = int(parts[2])
                
                if query_id not in relevance:
                    relevance[query_id] = {}
                relevance[query_id][doc_id] = score
    
    return relevance

