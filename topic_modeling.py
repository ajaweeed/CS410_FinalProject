"""
Topic modeling module using LDA and K-Means clustering
"""

from typing import List, Dict, Tuple
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from indexing import DocumentIndex
import config


class TopicModeler:
    """
    Topic modeling using LDA and K-Means clustering.
    """
    
    def __init__(self, index: DocumentIndex):
        """
        Initialize topic modeler.
        
        Args:
            index: DocumentIndex object
        """
        self.index = index
        self.lda_model = None
        self.kmeans_model = None
        self.vectorizer = None
        self.document_topic_distributions = None
        self.cluster_assignments = None
    
    def fit_lda(self, n_topics: int = None, alpha: float = None, beta: float = None):
        """
        Fit LDA topic model to the documents.
        
        Args:
            n_topics: Number of topics
            alpha: LDA alpha parameter
            beta: LDA beta parameter (learning_decay in sklearn)
        """
        if n_topics is None:
            n_topics = config.LDA_NUM_TOPICS
        if alpha is None:
            alpha = config.LDA_ALPHA
        if beta is None:
            beta = config.LDA_BETA
        
        # Prepare documents as strings of tokens
        documents = [' '.join(tokens) for tokens in self.index.processed_docs]
        
        # Create count vectorizer
        self.vectorizer = CountVectorizer(
            vocabulary=list(self.index.vocabulary),
            tokenizer=lambda x: x.split(),
            lowercase=False
        )
        
        # Create document-term matrix
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            doc_topic_prior=alpha,
            topic_word_prior=beta,
            random_state=42,
            max_iter=20,
            n_jobs=-1
        )
        
        self.lda_model.fit(doc_term_matrix)
        
        # Get document-topic distributions
        self.document_topic_distributions = self.lda_model.transform(doc_term_matrix)
        
        print(f"LDA model fitted with {n_topics} topics")
    
    def get_topics(self, n_words: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Get top words for each topic.
        
        Args:
            n_words: Number of top words to return per topic
            
        Returns:
            List of topics, each topic is a list of (word, weight) tuples
        """
        if self.lda_model is None:
            raise ValueError("LDA model not fitted. Call fit_lda() first.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [(feature_names[i], topic[i]) for i in top_indices]
            topics.append(top_words)
        
        return topics
    
    def get_document_topics(self, doc_id: int) -> List[Tuple[int, float]]:
        """
        Get topic distribution for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of (topic_id, probability) tuples
        """
        if self.document_topic_distributions is None:
            raise ValueError("LDA model not fitted. Call fit_lda() first.")
        
        if doc_id >= len(self.document_topic_distributions):
            return []
        
        topic_dist = self.document_topic_distributions[doc_id]
        topics = [(i, prob) for i, prob in enumerate(topic_dist)]
        topics.sort(key=lambda x: x[1], reverse=True)
        
        return topics
    
    def fit_kmeans(self, n_clusters: int = None, random_state: int = None):
        """
        Fit K-Means clustering to documents.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random state for reproducibility
        """
        if n_clusters is None:
            n_clusters = config.KMEANS_N_CLUSTERS
        if random_state is None:
            random_state = config.KMEANS_RANDOM_STATE
        
        # Use TF-IDF vectors as features
        num_docs = self.index.get_num_documents()
        vectors = []
        
        for doc_id in range(num_docs):
            vector = self.index.get_document_vector(doc_id)
            # Convert to array format (sparse representation)
            vec_array = np.zeros(len(self.index.vocabulary))
            vocab_list = list(self.index.vocabulary)
            for i, term in enumerate(vocab_list):
                vec_array[i] = vector.get(term, 0.0)
            vectors.append(vec_array)
        
        vectors = np.array(vectors)
        
        # Fit K-Means
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_assignments = self.kmeans_model.fit_predict(vectors)
        
        print(f"K-Means model fitted with {n_clusters} clusters")
    
    def get_cluster_assignments(self) -> List[int]:
        """
        Get cluster assignments for all documents.
        
        Returns:
            List of cluster IDs for each document
        """
        if self.cluster_assignments is None:
            raise ValueError("K-Means model not fitted. Call fit_kmeans() first.")
        
        return self.cluster_assignments.tolist()
    
    def get_document_cluster(self, doc_id: int) -> int:
        """
        Get cluster assignment for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Cluster ID
        """
        if self.cluster_assignments is None:
            raise ValueError("K-Means model not fitted. Call fit_kmeans() first.")
        
        if doc_id >= len(self.cluster_assignments):
            return -1
        
        return int(self.cluster_assignments[doc_id])
    
    def get_cluster_documents(self, cluster_id: int) -> List[int]:
        """
        Get all documents in a cluster.
        
        Args:
            cluster_id: Cluster identifier
            
        Returns:
            List of document IDs in the cluster
        """
        if self.cluster_assignments is None:
            raise ValueError("K-Means model not fitted. Call fit_kmeans() first.")
        
        doc_ids = []
        for doc_id, cluster in enumerate(self.cluster_assignments):
            if cluster == cluster_id:
                doc_ids.append(doc_id)
        
        return doc_ids
    
    def cluster_search_results(self, search_results: List[Tuple[int, float]], 
                              method: str = "kmeans", n_clusters: int = None) -> Dict[int, List[Tuple[int, float]]]:
        """
        Cluster search results into topic groups.
        
        Args:
            search_results: List of (doc_id, score) tuples from search
            method: Clustering method ("kmeans" or "lda")
            n_clusters: Number of clusters (for kmeans, ignored for lda)
            
        Returns:
            Dictionary mapping cluster_id to list of (doc_id, score) tuples
        """
        if not search_results:
            return {}
        
        # Extract document IDs from search results
        doc_ids = [doc_id for doc_id, score in search_results]
        
        if method == "kmeans":
            if self.kmeans_model is None:
                if n_clusters is None:
                    n_clusters = min(config.KMEANS_N_CLUSTERS, len(search_results))
                self.fit_kmeans(n_clusters=n_clusters)
            
            # Group results by cluster
            clusters = {}
            for doc_id, score in search_results:
                cluster_id = self.get_document_cluster(doc_id)
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append((doc_id, score))
            
            return clusters
        
        elif method == "lda":
            if self.lda_model is None:
                self.fit_lda()
            
            # Group results by dominant topic
            clusters = {}
            for doc_id, score in search_results:
                topics = self.get_document_topics(doc_id)
                if topics:
                    dominant_topic = topics[0][0]  # Get topic with highest probability
                    if dominant_topic not in clusters:
                        clusters[dominant_topic] = []
                    clusters[dominant_topic].append((doc_id, score))
                else:
                    # Assign to topic 0 if no topics found
                    if 0 not in clusters:
                        clusters[0] = []
                    clusters[0].append((doc_id, score))
            
            return clusters
        
        else:
            raise ValueError(f"Unknown clustering method: {method}")




