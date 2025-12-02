"""
Evaluation metrics for the search engine
"""

from typing import List, Dict, Tuple
import numpy as np
from ranking import BM25Ranker
from indexing import DocumentIndex
from query_expansion import QueryExpander
import config


class SearchEvaluator:
    """
    Evaluation metrics for search engine performance.
    """
    
    def __init__(self, index: DocumentIndex, ranker: BM25Ranker, 
                 query_expander: QueryExpander = None):
        """
        Initialize evaluator.
        
        Args:
            index: DocumentIndex object
            ranker: BM25Ranker object
            query_expander: QueryExpander object (optional)
        """
        self.index = index
        self.ranker = ranker
        self.query_expander = query_expander
    
    def precision_at_k(self, retrieved_doc_ids: List[int], 
                      relevant_doc_ids: List[int], k: int) -> float:
        """
        Calculate Precision@k.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_doc_ids: Set of relevant document IDs
            k: Top k results to consider
            
        Returns:
            Precision@k score
        """
        if k == 0:
            return 0.0
        
        relevant_set = set(relevant_doc_ids)
        retrieved_top_k = retrieved_doc_ids[:k]
        
        if not retrieved_top_k:
            return 0.0
        
        relevant_retrieved = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set)
        
        return relevant_retrieved / len(retrieved_top_k)
    
    def recall_at_k(self, retrieved_doc_ids: List[int], 
                   relevant_doc_ids: List[int], k: int) -> float:
        """
        Calculate Recall@k.
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevant_doc_ids: Set of relevant document IDs
            k: Top k results to consider
            
        Returns:
            Recall@k score
        """
        relevant_set = set(relevant_doc_ids)
        
        if not relevant_set:
            return 0.0
        
        retrieved_top_k = retrieved_doc_ids[:k]
        relevant_retrieved = sum(1 for doc_id in retrieved_top_k if doc_id in relevant_set)
        
        return relevant_retrieved / len(relevant_set)
    
    def ndcg_at_k(self, retrieved_doc_ids: List[int], 
                  relevance_scores: Dict[int, int], k: int) -> float:
        """
        Calculate NDCG@k (Normalized Discounted Cumulative Gain).
        
        Args:
            retrieved_doc_ids: List of retrieved document IDs
            relevance_scores: Dictionary mapping doc_id to relevance score
            k: Top k results to consider
            
        Returns:
            NDCG@k score
        """
        if k == 0:
            return 0.0
        
        retrieved_top_k = retrieved_doc_ids[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved_top_k):
            relevance = relevance_scores.get(doc_id, 0)
            if relevance > 0:
                # DCG: relevance / log2(i + 2) where i is 0-indexed
                dcg += relevance / np.log2(i + 2)
        
        # Calculate ideal DCG (IDCG)
        ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances) if rel > 0)
        
        # NDCG = DCG / IDCG
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_query(self, query: str, relevant_doc_ids: List[int], 
                      relevance_scores: Dict[int, int] = None,
                      use_expansion: bool = True, 
                      top_k: int = None) -> Dict[str, float]:
        """
        Evaluate a single query.
        
        Args:
            query: Query string
            relevant_doc_ids: List of relevant document IDs
            relevance_scores: Dictionary mapping doc_id to relevance score (for NDCG)
            use_expansion: Whether to use query expansion
            top_k: Maximum number of results to retrieve
            
        Returns:
            Dictionary of evaluation metrics
        """
        if top_k is None:
            top_k = max(config.EVAL_TOP_K)
        
        # Get search results
        if use_expansion and self.query_expander:
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
        
        retrieved_doc_ids = [doc_id for doc_id, score in results]
        
        # Calculate metrics
        metrics = {}
        
        for k in config.EVAL_TOP_K:
            metrics[f'precision@{k}'] = self.precision_at_k(
                retrieved_doc_ids, relevant_doc_ids, k
            )
            metrics[f'recall@{k}'] = self.recall_at_k(
                retrieved_doc_ids, relevant_doc_ids, k
            )
        
        # Calculate NDCG
        if relevance_scores:
            metrics[f'ndcg@{config.NDCG_K}'] = self.ndcg_at_k(
                retrieved_doc_ids, relevance_scores, config.NDCG_K
            )
        else:
            # Convert relevant_doc_ids to binary relevance scores
            binary_scores = {doc_id: 1 for doc_id in relevant_doc_ids}
            metrics[f'ndcg@{config.NDCG_K}'] = self.ndcg_at_k(
                retrieved_doc_ids, binary_scores, config.NDCG_K
            )
        
        return metrics
    
    def evaluate_queries(self, queries: List[str], 
                        relevance_judgments: Dict[int, Dict[int, int]],
                        use_expansion: bool = True,
                        top_k: int = None) -> Dict[str, float]:
        """
        Evaluate multiple queries and return average metrics.
        
        Args:
            queries: List of query strings
            relevance_judgments: Dictionary mapping query_id to doc_id to relevance_score
            use_expansion: Whether to use query expansion
            top_k: Maximum number of results to retrieve per query
            
        Returns:
            Dictionary of average evaluation metrics
        """
        all_metrics = []
        
        for query_id, query in enumerate(queries):
            if query_id not in relevance_judgments:
                continue
            
            relevance_dict = relevance_judgments[query_id]
            relevant_doc_ids = [doc_id for doc_id, score in relevance_dict.items() if score > 0]
            
            if not relevant_doc_ids:
                continue
            
            metrics = self.evaluate_query(
                query,
                relevant_doc_ids,
                relevance_scores=relevance_dict,
                use_expansion=use_expansion,
                top_k=top_k
            )
            
            all_metrics.append(metrics)
        
        if not all_metrics:
            return {}
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
            avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])
        
        return avg_metrics
    
    def print_evaluation_results(self, metrics: Dict[str, float], query_id: int = None):
        """
        Print evaluation results in a readable format.
        
        Args:
            metrics: Dictionary of evaluation metrics
            query_id: Query ID (optional, for single query evaluation)
        """
        if query_id is not None:
            print(f"\nEvaluation results for Query {query_id}:")
        else:
            print("\nAverage Evaluation Results:")
        
        print("-" * 50)
        
        for k in config.EVAL_TOP_K:
            prec_key = f'precision@{k}'
            rec_key = f'recall@{k}'
            
            if prec_key in metrics:
                print(f"Precision@{k}: {metrics[prec_key]:.4f}")
            if rec_key in metrics:
                print(f"Recall@{k}: {metrics[rec_key]:.4f}")
        
        ndcg_key = f'ndcg@{config.NDCG_K}'
        if ndcg_key in metrics:
            print(f"NDCG@{config.NDCG_K}: {metrics[ndcg_key]:.4f}")
        
        print("-" * 50)




