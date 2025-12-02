"""
Streamlit web interface for the Topic-Aware Search Engine
"""

import streamlit as st
import pandas as pd
from search_engine import TopicAwareSearchEngine
from utils import load_documents, load_queries, load_relevance_judgments
import config

# Page configuration
st.set_page_config(
    page_title="Topic-Aware Search Engine",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = None
if 'index_built' not in st.session_state:
    st.session_state.index_built = False


def load_data_and_build_index(data_path: str):
    """Load documents and build index."""
    try:
        with st.spinner("Loading documents and building index..."):
            engine = TopicAwareSearchEngine(data_path=data_path)
            engine.build_index()
            st.session_state.search_engine = engine
            st.session_state.index_built = True
            st.success("Index built successfully!")
            return True
    except Exception as e:
        st.error(f"Error building index: {str(e)}")
        return False


def main():
    st.title("🔍 Topic-Aware Search Engine with Query Expansion")
    st.markdown("CS410 Fall 2025 - Information Retrieval Project")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Data loading
        st.subheader("Data Loading")
        data_path = st.text_input(
            "Document File Path",
            value=config.DEFAULT_DATA_PATH,
            help="Path to the file containing documents (one per line)"
        )
        
        if st.button("Load Documents & Build Index"):
            if load_data_and_build_index(data_path):
                st.rerun()
        
        if st.session_state.index_built:
            st.success("✓ Index ready")
            
            # Display statistics
            stats = st.session_state.search_engine.get_statistics()
            st.subheader("Statistics")
            st.write(f"Documents: {stats['num_documents']}")
            st.write(f"Vocabulary: {stats['vocabulary_size']}")
            st.write(f"Avg Doc Length: {stats['avg_doc_length']:.1f}")
        
        # Search configuration
        st.subheader("Search Settings")
        use_expansion = st.checkbox("Use Query Expansion", value=True)
        expansion_method = st.selectbox(
            "Expansion Method",
            ["combined", "wordnet", "embedding"],
            help="Method for query expansion"
        )
        top_k = st.slider("Number of Results", 5, 50, 10)
        
        # Topic modeling settings
        st.subheader("Topic Modeling")
        cluster_method = st.selectbox(
            "Clustering Method",
            ["kmeans", "lda"],
            help="Method for clustering search results"
        )
        n_clusters = st.slider("Number of Clusters", 2, 10, 5)
        
        # Fit topic models button
        if st.session_state.index_built and st.button("Fit Topic Models"):
            with st.spinner("Fitting topic models..."):
                try:
                    st.session_state.search_engine.fit_topic_models(
                        n_topics=config.LDA_NUM_TOPICS,
                        n_clusters=n_clusters
                    )
                    st.success("Topic models fitted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error fitting topic models: {str(e)}")
    
    # Main content area
    if not st.session_state.index_built:
        st.info("👈 Please load documents and build the index using the sidebar.")
        st.markdown("""
        ### Instructions:
        1. Enter the path to your document file in the sidebar
        2. Click "Load Documents & Build Index"
        3. Wait for the index to be built
        4. Start searching!
        """)
    else:
        # Tabs for different functionalities
        tab1, tab2, tab3, tab4 = st.tabs(["Search", "Topic Clusters", "Topics", "Evaluation"])
        
        # Search tab
        with tab1:
            st.header("Search")
            query = st.text_input("Enter your query:", "")
            
            if query:
                col1, col2 = st.columns([3, 1])
                with col1:
                    search_button = st.button("Search", type="primary")
                with col2:
                    show_expansion = st.checkbox("Show Expansion Terms", value=False)
                
                if search_button:
                    with st.spinner("Searching..."):
                        try:
                            # Get expansion terms if requested
                            if show_expansion:
                                expansion_terms = st.session_state.search_engine.get_expansion_terms(query)
                                if expansion_terms:
                                    st.subheader("Query Expansion Terms")
                                    expansion_df = pd.DataFrame([
                                        {"Term": term, "Score": score}
                                        for term, score in sorted(
                                            expansion_terms.items(),
                                            key=lambda x: x[1],
                                            reverse=True
                                        )[:10]
                                    ])
                                    st.dataframe(expansion_df, use_container_width=True)
                            
                            # Perform search
                            results = st.session_state.search_engine.search(
                                query,
                                top_k=top_k,
                                use_expansion=use_expansion,
                                expansion_method=expansion_method
                            )
                            
                            if results:
                                st.subheader(f"Search Results ({len(results)} documents)")
                                
                                for i, (doc_id, score, doc_text) in enumerate(results, 1):
                                    with st.expander(f"Result {i} - Doc ID: {doc_id} (Score: {score:.4f})"):
                                        st.write(doc_text)
                            else:
                                st.warning("No results found.")
                        
                        except Exception as e:
                            st.error(f"Error during search: {str(e)}")
        
        # Topic Clusters tab
        with tab2:
            st.header("Topic Clusters")
            cluster_query = st.text_input("Enter query for clustering:", key="cluster_query")
            
            if cluster_query and st.button("Cluster Results", key="cluster_button"):
                with st.spinner("Clustering results..."):
                    try:
                        # Get search results
                        results = st.session_state.search_engine.search(
                            cluster_query,
                            top_k=top_k,
                            use_expansion=use_expansion
                        )
                        
                        if results:
                            # Get clusters
                            clusters = st.session_state.search_engine.get_topic_clusters(
                                results,
                                method=cluster_method,
                                n_clusters=n_clusters
                            )
                            
                            st.subheader(f"Topic Clusters ({len(clusters)} clusters)")
                            
                            for cluster_id, cluster_results in clusters.items():
                                st.markdown(f"### Cluster {cluster_id} ({len(cluster_results)} documents)")
                                
                                for doc_id, score, doc_text in cluster_results:
                                    with st.expander(f"Doc ID: {doc_id} (Score: {score:.4f})"):
                                        st.write(doc_text)
                        else:
                            st.warning("No results to cluster.")
                    
                    except Exception as e:
                        st.error(f"Error during clustering: {str(e)}")
        
        # Topics tab
        with tab3:
            st.header("Topic Modeling")
            
            if st.button("Show Topics"):
                try:
                    stats = st.session_state.search_engine.get_statistics()
                    if not stats['lda_fitted']:
                        with st.spinner("Fitting LDA model..."):
                            st.session_state.search_engine.fit_topic_models(
                                n_topics=config.LDA_NUM_TOPICS
                            )
                    
                    topics = st.session_state.search_engine.get_topics(n_words=10)
                    
                    st.subheader(f"LDA Topics ({len(topics)} topics)")
                    
                    for topic_id, topic_words in enumerate(topics):
                        st.markdown(f"### Topic {topic_id}")
                        words_text = ", ".join([f"{word} ({weight:.3f})" for word, weight in topic_words])
                        st.write(words_text)
                
                except Exception as e:
                    st.error(f"Error displaying topics: {str(e)}")
        
        # Evaluation tab
        with tab4:
            st.header("Evaluation")
            
            queries_path = st.text_input(
                "Queries File Path",
                value=config.DEFAULT_QUERIES_PATH,
                key="queries_path"
            )
            relevance_path = st.text_input(
                "Relevance Judgments File Path",
                value=config.DEFAULT_RELEVANCE_PATH,
                key="relevance_path"
            )
            
            if st.button("Run Evaluation"):
                try:
                    with st.spinner("Running evaluation..."):
                        queries = load_queries(queries_path)
                        relevance_judgments = load_relevance_judgments(relevance_path)
                        
                        # Convert query indices (0-based)
                        relevance_dict = {
                            i: relevance_judgments.get(i + 1, {})
                            for i in range(len(queries))
                        }
                        
                        metrics = st.session_state.search_engine.evaluate(
                            queries,
                            relevance_dict,
                            use_expansion=use_expansion
                        )
                        
                        if metrics:
                            st.subheader("Evaluation Results")
                            
                            # Create metrics DataFrame
                            metric_names = [k for k in metrics.keys() if not k.endswith('_std')]
                            metric_values = [metrics[k] for k in metric_names]
                            
                            metrics_df = pd.DataFrame({
                                "Metric": metric_names,
                                "Value": metric_values
                            })
                            
                            st.dataframe(metrics_df, use_container_width=True)
                            
                            # Display as bar chart
                            st.bar_chart(metrics_df.set_index("Metric"))
                        else:
                            st.warning("No evaluation metrics computed.")
                
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")


if __name__ == "__main__":
    main()




