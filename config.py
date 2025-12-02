"""
Configuration file for the Topic-Aware Search Engine
"""

# BM25 Parameters
BM25_K1 = 1.5  # Term frequency saturation parameter
BM25_B = 0.75  # Length normalization parameter

# Query Expansion Parameters
WORDNET_WEIGHT = 0.5  # Weight for WordNet expansion
EMBEDDING_WEIGHT = 0.5  # Weight for embedding-based expansion
EXPANSION_TOP_K = 5  # Number of expansion terms to add
EMBEDDING_SIMILARITY_THRESHOLD = 0.6  # Minimum similarity for embedding expansion

# Topic Modeling Parameters
LDA_NUM_TOPICS = 10  # Number of topics for LDA
LDA_ALPHA = 0.1  # LDA alpha parameter
LDA_BETA = 0.01  # LDA beta parameter
KMEANS_N_CLUSTERS = 5  # Number of clusters for K-Means
KMEANS_RANDOM_STATE = 42  # Random state for reproducibility

# Embedding Parameters
GLOVE_DIM = 100  # GloVe embedding dimension (50, 100, 200, or 300)
GLOVE_URL = "https://nlp.stanford.edu/data/glove.6B.zip"  # GloVe download URL
EMBEDDINGS_CACHE_DIR = "embeddings"  # Directory to cache embeddings

# Indexing Parameters
MIN_TERM_LENGTH = 2  # Minimum term length
MAX_DF = 0.95  # Maximum document frequency (ignore terms in >95% of documents)
MIN_DF = 1  # Minimum document frequency (ignore terms in <1 document)

# Evaluation Parameters
EVAL_TOP_K = [5, 10, 20]  # Top-k values for evaluation metrics
NDCG_K = 10  # NDCG@k value

# Streamlit Interface Parameters
RESULTS_PER_PAGE = 10  # Number of results to display per page
MAX_QUERY_LENGTH = 500  # Maximum query length

# Data Paths
DEFAULT_DATA_PATH = "data/documents.txt"
DEFAULT_QUERIES_PATH = "data/queries.txt"
DEFAULT_RELEVANCE_PATH = "data/relevance.txt"




