"""
Evaluation script for the search engine
"""

import argparse
from search_engine import TopicAwareSearchEngine
from utils import load_documents, load_queries, load_relevance_judgments
import config


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Topic-Aware Search Engine"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.DEFAULT_DATA_PATH,
        help="Path to document file"
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
        "--top_k",
        type=int,
        default=None,
        help="Maximum number of results to retrieve per query"
    )
    
    args = parser.parse_args()
    
    # Initialize search engine
    print("Initializing search engine...")
    engine = TopicAwareSearchEngine(data_path=args.data_path)
    
    # Build index
    print("Building index...")
    engine.build_index()
    
    # Load queries and relevance judgments
    print("Loading queries and relevance judgments...")
    queries = load_queries(args.queries_path)
    relevance_judgments = load_relevance_judgments(args.relevance_path)
    
    # Convert query indices (assuming 1-based in file, 0-based in code)
    relevance_dict = {
        i: relevance_judgments.get(i + 1, {})
        for i in range(len(queries))
    }
    
    # Run evaluation
    print("Running evaluation...")
    metrics = engine.evaluate(
        queries,
        relevance_dict,
        use_expansion=args.use_expansion,
        top_k=args.top_k
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    if args.use_expansion:
        print("Query Expansion: ENABLED")
    else:
        print("Query Expansion: DISABLED")
    
    print("\nAverage Metrics:")
    print("-" * 60)
    
    for metric, value in sorted(metrics.items()):
        if not metric.endswith('_std'):
            std_key = f"{metric}_std"
            if std_key in metrics:
                print(f"{metric:20s}: {value:8.4f} ± {metrics[std_key]:.4f}")
            else:
                print(f"{metric:20s}: {value:8.4f}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()




