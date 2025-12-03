# CS410_FinalProject


# Topic-Aware Search Engine with Query Expansion

A semantic search engine that enhances retrieval quality through query expansion and topic clustering, built for CS410 Fall 2025.

## Project Overview

This project implements a topic-aware search engine that addresses the challenge of matching user queries with documents even when exact phrasing differs. The system combines:

- **Semantic Query Expansion**: Uses WordNet synonyms and GloVe embeddings to expand queries
- **BM25 Ranking**: Implements the BM25 algorithm for document ranking
- **Topic Clustering**: Organizes results using LDA topic modeling and K-Means clustering
- **Evaluation Metrics**: Measures precision, recall, and NDCG@k

## Features

- Document indexing with tokenization, stopword removal, and term weighting
- Query expansion using WordNet and embedding-based similarity
- BM25-based document ranking
- Topic modeling with LDA and K-Means clustering
- Web interface built with Streamlit
- Comprehensive evaluation metrics

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd CS410
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"
```

4. Download GloVe embeddings (optional, will be downloaded automatically):
   - The system will automatically download GloVe embeddings on first run
   - Alternatively, download manually from: https://nlp.stanford.edu/projects/glove/

## Usage

### Command Line Interface

```bash
# Index documents and run search
python main.py --data_path data/documents.txt --query "your query here"

# Run evaluation
python evaluate.py --data_path data/documents.txt --queries data/queries.txt --relevance data/relevance.txt
```

### Web Interface

```bash
python -m streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Python API

```python
from search_engine import TopicAwareSearchEngine

# Initialize the search engine
engine = TopicAwareSearchEngine(data_path="data/documents.txt")

# Build index
engine.build_index()

# Search
results = engine.search("machine learning algorithms", top_k=10)

# Get topic clusters
clusters = engine.get_topic_clusters(results, n_clusters=3)
```

## Project Structure

```
CS410/
├── app.py                 # Streamlit web interface
├── main.py               # Main entry point
├── search_engine.py      # Core search engine class
├── indexing.py           # Document indexing module
├── query_expansion.py    # Query expansion module
├── ranking.py            # BM25 ranking implementation
├── topic_modeling.py     # LDA and K-Means clustering
├── evaluation.py         # Evaluation metrics
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Data directory
│   ├── documents.txt    # Sample documents
│   ├── queries.txt      # Sample queries
│   └── relevance.txt    # Relevance judgments
└── docs/                # Documentation
    └── implementation.md # Implementation details
```

## Data Format

### Documents
Each document should be on a separate line in a text file:
```
Document 1 text content here...
Document 2 text content here...
```

### Queries
One query per line:
```
query 1
query 2
```

### Relevance Judgments
Tab-separated format: `query_id\tdoc_id\trelevance_score`
```
1	0	1
1	5	2
2	3	1
```

## Implementation Details

### Document Indexing
- Tokenization using NLTK
- Stopword removal (English)
- Term frequency-inverse document frequency (TF-IDF) weighting
- Vocabulary building and term-document matrix construction

### Query Expansion
- **WordNet Expansion**: Extracts synonyms from WordNet
- **Embedding Expansion**: Uses GloVe embeddings to find semantically similar terms
- Combines both methods with configurable weights

### BM25 Ranking
- Implements the BM25 algorithm with configurable parameters (k1=1.5, b=0.75)
- Handles term frequency and document length normalization

### Topic Modeling
- **LDA**: Latent Dirichlet Allocation for topic discovery
- **K-Means**: Clustering based on document vectors
- Both methods can be used to group search results

### Evaluation
- **Precision@k**: Fraction of retrieved documents that are relevant
- **Recall@k**: Fraction of relevant documents that are retrieved
- **NDCG@k**: Normalized Discounted Cumulative Gain

## Configuration

Edit `config.py` to customize:
- BM25 parameters (k1, b)
- Query expansion weights
- Number of topics for LDA
- Number of clusters for K-Means
- Embedding dimensions

## Testing

Run the test suite:
```bash
pytest tests/
```

## Evaluation Results

Evaluation results will be displayed when running `evaluate.py` or through the web interface's evaluation tab.

## Contributing

This is a course project for CS410 Fall 2025. For questions or issues, please contact the project coordinator.

## License

This project is for educational purposes as part of CS410 Fall 2025.

## Team Members

- Afraz Jaweed (jaweeed2) - Project Coordinator

## References

- BM25: Robertson, S. E., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond.
- LDA: Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent dirichlet allocation.
- GloVe: Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global vectors for word representation.



