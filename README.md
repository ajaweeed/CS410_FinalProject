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
