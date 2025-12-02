"""
Main search engine class that orchestrates all components
"""

from typing import List, Dict, Tuple, Optional
from indexing import DocumentIndex
from query_expansion import QueryExpander
from ranking import BM25Ranker
from topic_modeling import TopicModeler
from evaluation import SearchEvaluator
from utils import load_documents
import config


class TopicAwareSearchEngine:
    """
    Topic-aware search engine with query expansion and topic clustering.
    """
    
    def __init__(self, data_path: str = None, documents: List[str] = None,
                 embedding_dim: int = None):
        """
        Initialize the search engine.
        
        Args:
            data_path: Path to document file (optional)
            documents: List of document strings (optional)
            embedding_dim: GloVe embedding dimension
        """
        self.index = DocumentIndex()
        self.query_expander = QueryExpander(embedding_dim=embedding_dim)
        self.ranker = None
        self.topic_modeler = None
        self.evaluator = None
        
        # Load documents if provided
        if data_path:
            documents = load_documents(data_path)
        
        if documents:
            self.documents = documents
        else:
            self.documents = []
    
    def build_index(self, documents: List[str] = None):
        """
        Build the document index.
        
        Args:
            documents: List of document strings (optional, uses self.documents if not provided)
        """
        if documents:
            self.documents = documents
        
        if not self.documents:
            raise ValueError("No documents provided. Load documents first.")
        
        print(f"Building index for {len(self.documents)} documents...")
        self.index.build_index(self.documents)
        
        # Initialize ranker and topic modeler
        self.ranker = BM25Ranker(self.index)
        self.topic_modeler = TopicModeler(self.index)
        self.evaluator = SearchEvaluator(self.index, self.ranker, self.query_expander)
        
        print(f"Index built. Vocabulary size: {self.index.get_vocabulary_size()}")
        print(f"Average document length: {self.index.avg_doc_length:.2f}")
    
    def search(self, query: str, top_k: int = 10, use_expansion: bool = True,
              expansion_method: str = "combined") -> List[Tuple[int, float, str]]:
        """
        Search for documents matching a query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            use_expansion: Whether to use query expansion
            expansion_method: Expansion method ("wordnet", "embedding", or "combined")
            
        Returns:
            List of (doc_id, score, document_text) tuples
        """
        if self.ranker is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Rank documents
        if use_expansion:
            results = self.ranker.rank_documents(
                query,
                query_expander=self.query_expander,
                expanded=True,
                top_k=top_k
            )
        else:
            results = self.ranker.rank_documents(
                query,
                expanded=False,
                top_k=top_k
            )
        
        # Format results
        formatted_results = []
        for doc_id, score in results:
            doc_text = self.index.documents[doc_id] if doc_id < len(self.index.documents) else ""
            formatted_results.append((doc_id, score, doc_text))
        
        return formatted_results
    
    def get_topic_clusters(self, search_results: List[Tuple[int, float, str]],
                          method: str = "kmeans", n_clusters: int = None) -> Dict[int, List[Tuple[int, float, str]]]:
        """
        Cluster search results into topic groups.
        
        Args:
            search_results: List of (doc_id, score, document_text) tuples
            method: Clustering method ("kmeans" or "lda")
            n_clusters: Number of clusters (for kmeans, ignored for lda)
            
        Returns:
            Dictionary mapping cluster_id to list of (doc_id, score, document_text) tuples
        """
        if self.topic_modeler is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Extract (doc_id, score) tuples for clustering
        results_for_clustering = [(doc_id, score) for doc_id, score, _ in search_results]
        
        # Get clusters
        clusters = self.topic_modeler.cluster_search_results(
            results_for_clustering,
            method=method,
            n_clusters=n_clusters
        )
        
        # Format clusters with document text
        formatted_clusters = {}
        for cluster_id, cluster_results in clusters.items():
            formatted_clusters[cluster_id] = [
                (doc_id, score, self.index.documents[doc_id] if doc_id < len(self.index.documents) else "")
                for doc_id, score in cluster_results
            ]
        
        return formatted_clusters
    
    def get_topics(self, n_topics: int = None, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get topics from LDA model.
        
        Args:
            n_topics: Number of topics (optional, uses config if not provided)
            n_words: Number of top words per topic
            
        Returns:
            List of topics, each topic is a list of (word, weight) tuples
        """
        if self.topic_modeler is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        if self.topic_modeler.lda_model is None:
            self.topic_modeler.fit_lda(n_topics=n_topics)
        
        return self.topic_modeler.get_topics(n_words=n_words)
    
    def fit_topic_models(self, n_topics: int = None, n_clusters: int = None):
        """
        Fit both LDA and K-Means topic models.
        
        Args:
            n_topics: Number of topics for LDA
            n_clusters: Number of clusters for K-Means
        """
        if self.topic_modeler is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        print("Fitting LDA model...")
        self.topic_modeler.fit_lda(n_topics=n_topics)
        
        print("Fitting K-Means model...")
        self.topic_modeler.fit_kmeans(n_clusters=n_clusters)
        
        print("Topic models fitted successfully.")
    
    def evaluate(self, queries: List[str], relevance_judgments: Dict[int, Dict[int, int]],
                use_expansion: bool = True, top_k: int = None) -> Dict[str, float]:
        """
        Evaluate search engine performance.
        
        Args:
            queries: List of query strings
            relevance_judgments: Dictionary mapping query_id to doc_id to relevance_score
            use_expansion: Whether to use query expansion
            top_k: Maximum number of results to retrieve per query
            
        Returns:
            Dictionary of average evaluation metrics
        """
        if self.evaluator is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        return self.evaluator.evaluate_queries(
            queries,
            relevance_judgments,
            use_expansion=use_expansion,
            top_k=top_k
        )
    
    def get_expansion_terms(self, query: str) -> Dict[str, float]:
        """
        Get query expansion terms with their scores.
        
        Args:
            query: Query string
            
        Returns:
            Dictionary mapping expansion terms to their scores
        """
        return self.query_expander.get_expansion_scores(query)
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get search engine statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'num_documents': self.index.get_num_documents(),
            'vocabulary_size': self.index.get_vocabulary_size(),
            'avg_doc_length': self.index.avg_doc_length,
            'index_built': self.ranker is not None,
            'lda_fitted': self.topic_modeler.lda_model is not None if self.topic_modeler else False,
            'kmeans_fitted': self.topic_modeler.kmeans_model is not None if self.topic_modeler else False,
        }
        
        return stats




