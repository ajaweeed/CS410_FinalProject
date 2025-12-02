"""
Main entry point for the Topic-Aware Search Engine
"""

import argparse
import sys
from search_engine import TopicAwareSearchEngine
from utils import load_documents, load_queries, load_relevance_judgments
from evaluation import SearchEvaluator
import config


def main():
    parser = argparse.ArgumentParser(
        description="Topic-Aware Search Engine with Query Expansion"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.DEFAULT_DATA_PATH,
        help="Path to document file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query string to search"
    )
    parser.add_argument(
        "--queries_path",
        type=str,
        default=config.DEFAULT_QUERIES_PATH,
        help="Path to queries file"
    )
    parser.add_argument(
        "--relevance_path",
        type=str,
        default=config.DEFAULT_RELEVANCE_PATH,
        help="Path to relevance judgments file"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top results to return"
    )
    parser.add_argument(
        "--use_expansion",
        action="store_true",
        default=True,
        help="Use query expansion"
    )
    parser.add_argument(
        "--no_expansion",
        action="store_false",
        dest="use_expansion",
        help="Do not use query expansion"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run evaluation"
    )
    parser.add_argument(
        "--expansion_method",
        type=str,
        default="combined",
        choices=["wordnet", "embedding", "combined"],
        help="Query expansion method"
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Cluster search results"
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "lda"],
        help="Clustering method"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters"
    )
    parser.add_argument(
        "--show_expansion",
        action="store_true",
        help="Show query expansion terms"
    )
    
    args = parser.parse_args()
    
    # Initialize search engine
    print("Initializing search engine...")
    engine = TopicAwareSearchEngine(data_path=args.data_path)
    
    # Build index
    print("Building index...")
    engine.build_index()
    
    # Run evaluation if requested
    if args.evaluate:
        print("Running evaluation...")
        try:
            queries = load_queries(args.queries_path)
            relevance_judgments = load_relevance_judgments(args.relevance_path)
            
            # Convert query indices (assuming 1-based in file, 0-based in code)
            relevance_dict = {
                i: relevance_judgments.get(i + 1, {})
                for i in range(len(queries))
            }
            
            metrics = engine.evaluate(
                queries,
                relevance_dict,
                use_expansion=args.use_expansion
            )
            
            print("\nEvaluation Results:")
            print("=" * 50)
            for metric, value in metrics.items():
                if not metric.endswith('_std'):
                    std_key = f"{metric}_std"
                    if std_key in metrics:
                        print(f"{metric}: {value:.4f} ± {metrics[std_key]:.4f}")
                    else:
                        print(f"{metric}: {value:.4f}")
            print("=" * 50)
        
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            sys.exit(1)
    
    # Perform search if query provided
    elif args.query:
        print(f"Searching for: {args.query}")
        
        # Show expansion terms if requested
        if args.show_expansion:
            expansion_terms = engine.get_expansion_terms(args.query)
            print("\nQuery Expansion Terms:")
            print("-" * 50)
            for term, score in sorted(expansion_terms.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"{term}: {score:.4f}")
            print("-" * 50)
        
        # Perform search
        results = engine.search(
            args.query,
            top_k=args.top_k,
            use_expansion=args.use_expansion,
            expansion_method=args.expansion_method
        )
        
        if results:
            print(f"\nSearch Results ({len(results)} documents):")
            print("=" * 50)
            
            for i, (doc_id, score, doc_text) in enumerate(results, 1):
                print(f"\nResult {i} - Doc ID: {doc_id} (Score: {score:.4f})")
                print("-" * 50)
                # Show first 200 characters of document
                preview = doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                print(preview)
            
            # Cluster results if requested
            if args.cluster:
                print("\n" + "=" * 50)
                print("Clustering Results...")
                clusters = engine.get_topic_clusters(
                    results,
                    method=args.cluster_method,
                    n_clusters=args.n_clusters
                )
                
                print(f"\nTopic Clusters ({len(clusters)} clusters):")
                print("=" * 50)
                
                for cluster_id, cluster_results in clusters.items():
                    print(f"\nCluster {cluster_id} ({len(cluster_results)} documents):")
                    print("-" * 50)
                    for doc_id, score, doc_text in cluster_results:
                        preview = doc_text[:100] + "..." if len(doc_text) > 100 else doc_text
                        print(f"  Doc ID: {doc_id} (Score: {score:.4f}) - {preview}")
        else:
            print("No results found.")
    
    else:
        print("No query provided. Use --query to search or --evaluate to run evaluation.")
        print("Use --help for more information.")


if __name__ == "__main__":
    main()




