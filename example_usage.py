"""
Example usage of the Topic-Aware Search Engine
"""

from search_engine import TopicAwareSearchEngine
from utils import load_documents, load_queries, load_relevance_judgments
import config


def example_basic_search():
    """Example: Basic search functionality"""
    print("=" * 60)
    print("Example 1: Basic Search")
    print("=" * 60)
    
    # Initialize search engine
    engine = TopicAwareSearchEngine(data_path="data/documents.txt")
    
    # Build index
    engine.build_index()
    
    # Perform search
    query = "machine learning"
    results = engine.search(query, top_k=5, use_expansion=True)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")
    
    for i, (doc_id, score, doc_text) in enumerate(results, 1):
        print(f"{i}. Doc ID: {doc_id} (Score: {score:.4f})")
        print(f"   {doc_text[:100]}...")
        print()


def example_query_expansion():
    """Example: Query expansion"""
    print("=" * 60)
    print("Example 2: Query Expansion")
    print("=" * 60)
    
    # Initialize search engine
    engine = TopicAwareSearchEngine(data_path="data/documents.txt")
    engine.build_index()
    
    # Get expansion terms
    query = "information retrieval"
    expansion_terms = engine.get_expansion_terms(query)
    
    print(f"\nQuery: {query}")
    print(f"Expansion terms: {len(expansion_terms)}")
    print("\nTop expansion terms:")
    for term, score in sorted(expansion_terms.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {term}: {score:.4f}")


def example_topic_clustering():
    """Example: Topic clustering of search results"""
    print("=" * 60)
    print("Example 3: Topic Clustering")
    print("=" * 60)
    
    # Initialize search engine
    engine = TopicAwareSearchEngine(data_path="data/documents.txt")
    engine.build_index()
    
    # Fit topic models
    engine.fit_topic_models(n_topics=5, n_clusters=3)
    
    # Perform search
    query = "topic modeling"
    results = engine.search(query, top_k=10, use_expansion=True)
    
    # Cluster results
    clusters = engine.get_topic_clusters(results, method="kmeans", n_clusters=3)
    
    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results")
    print(f"Clustered into {len(clusters)} topic clusters:\n")
    
    for cluster_id, cluster_results in clusters.items():
        print(f"Cluster {cluster_id} ({len(cluster_results)} documents):")
        for doc_id, score, doc_text in cluster_results:
            print(f"  - Doc {doc_id}: {doc_text[:60]}...")
        print()


def example_topic_modeling():
    """Example: Topic modeling with LDA"""
    print("=" * 60)
    print("Example 4: Topic Modeling (LDA)")
    print("=" * 60)
    
    # Initialize search engine
    engine = TopicAwareSearchEngine(data_path="data/documents.txt")
    engine.build_index()
    
    # Fit LDA model
    engine.fit_topic_models(n_topics=5)
    
    # Get topics
    topics = engine.get_topics(n_words=5)
    
    print(f"\nDiscovered {len(topics)} topics:\n")
    for topic_id, topic_words in enumerate(topics):
        print(f"Topic {topic_id}:")
        words = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_words])
        print(f"  {words}")
        print()


def example_evaluation():
    """Example: Evaluation with relevance judgments"""
    print("=" * 60)
    print("Example 5: Evaluation")
    print("=" * 60)
    
    # Initialize search engine
    engine = TopicAwareSearchEngine(data_path="data/documents.txt")
    engine.build_index()
    
    # Load queries and relevance judgments
    queries = load_queries("data/queries.txt")
    relevance_judgments = load_relevance_judgments("data/relevance.txt")
    
    # Convert query indices (assuming 1-based in file, 0-based in code)
    relevance_dict = {
        i: relevance_judgments.get(i + 1, {})
        for i in range(len(queries))
    }
    
    # Evaluate with expansion
    print("\nEvaluation with query expansion:")
    metrics_expanded = engine.evaluate(
        queries,
        relevance_dict,
        use_expansion=True
    )
    
    print_metrics(metrics_expanded)
    
    # Evaluate without expansion
    print("\nEvaluation without query expansion:")
    metrics_no_expansion = engine.evaluate(
        queries,
        relevance_dict,
        use_expansion=False
    )
    
    print_metrics(metrics_no_expansion)


def print_metrics(metrics):
    """Print evaluation metrics"""
    print("-" * 60)
    for metric, value in sorted(metrics.items()):
        if not metric.endswith('_std'):
            std_key = f"{metric}_std"
            if std_key in metrics:
                print(f"{metric:20s}: {value:8.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"{metric:20s}: {value:8.4f}")
    print("-" * 60)


def example_statistics():
    """Example: Get search engine statistics"""
    print("=" * 60)
    print("Example 6: Statistics")
    print("=" * 60)
    
    # Initialize search engine
    engine = TopicAwareSearchEngine(data_path="data/documents.txt")
    engine.build_index()
    
    # Get statistics
    stats = engine.get_statistics()
    
    print("\nSearch Engine Statistics:")
    print("-" * 60)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    print("-" * 60)


if __name__ == "__main__":
    # Run examples
    try:
        example_basic_search()
        print("\n")
        
        example_query_expansion()
        print("\n")
        
        example_topic_clustering()
        print("\n")
        
        example_topic_modeling()
        print("\n")
        
        example_evaluation()
        print("\n")
        
        example_statistics()
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        import traceback
        traceback.print_exc()




