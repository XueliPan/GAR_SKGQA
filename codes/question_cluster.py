"""
Question clustering script
--------------------------
Clustering ~2500 natural language questions by semantic meaning.

Features:
- Accepts a CSV or TXT file of questions
- Two embedding backends: sentence-transformers (local) or OpenAI (cloud)
- Dimensionality reduction with UMAP (optional)
- Clustering with HDBSCAN (recommended) or KMeans
- Produces: cluster labels, example questions per cluster, cluster sizes, and a CSV output
- Visualization (2D scatter) saved as PNG

Usage examples:
python question_clustering.py --input questions.csv --text_col question --backend st --model all-MiniLM-L6-v2
python question_clustering.py --input questions.txt --backend openai --openai_key YOUR_KEY

Requirements (pip install):
- sentence-transformers
- openai
- numpy
- pandas
- umap-learn
- hdbscan
- scikit-learn
- matplotlib

Notes:
- HDBSCAN discovers clusters and marks outliers as -1. It works well for varied cluster sizes.
- UMAP helps HDBSCAN when embeddings are high-dimensional (recommended).
- If using OpenAI embeddings, be mindful of rate limits and cost.

"""

import os
import argparse
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Embedding libs
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Optional OpenAI backend
try:
    import openai
except Exception:
    openai = None

# Dimensionality reduction & clustering
try:
    import umap
except Exception:
    umap = None

try:
    import hdbscan
except Exception:
    hdbscan = None

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def load_questions(path: str, text_col: Optional[str] = None) -> pd.DataFrame:
    """Load questions from CSV (with a text column) or plain TXT (one question per line).
    Returns a DataFrame with column 'question'.
    """
    path = os.path.expanduser(path)
    if path.endswith('.csv'):
        df = pd.read_csv(path)
        if text_col is None:
            # try to auto-detect
            candidates = [c for c in df.columns if 'question' in c.lower() or 'text' in c.lower() or 'query' in c.lower()]
            if candidates:
                text_col = candidates[0]
            else:
                raise ValueError('CSV input provided but --text_col not specified and could not auto-detect a column.')
        df = df[[text_col]].rename(columns={text_col: 'question'})
        df['question'] = df['question'].astype(str).str.strip()
    else:
        # treat as plain text file with one question per line
        with open(path, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        df = pd.DataFrame({'question': lines})
    df = df.drop_duplicates(subset=['question']).reset_index(drop=True)
    return df


def embed_with_st(model_name: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers is not installed. Install with `pip install sentence-transformers`.')
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)
    return embeddings


def embed_with_openai(api_key: str, model_name: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    if openai is None:
        raise RuntimeError('openai is not installed. Install with `pip install openai`.')
    openai.api_key = api_key
    embeddings = []
    # OpenAI rate-limits; do in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = openai.Embedding.create(input=batch, model=model_name)
        for r in response['data']:
            embeddings.append(r['embedding'])
    return np.array(embeddings)


def reduce_dim(embeddings: np.ndarray, n_components: int = 5, random_state: int = 42) -> np.ndarray:
    if umap is None:
        raise RuntimeError('umap-learn is not installed. Install with `pip install umap-learn`.')
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced = reducer.fit_transform(embeddings)
    return reduced


def cluster_hdbscan(embeddings: np.ndarray, min_cluster_size: int = 5, min_samples: Optional[int] = None) -> np.ndarray:
    if hdbscan is None:
        raise RuntimeError('hdbscan is not installed. Install with `pip install hdbscan`.')
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
    labels = clusterer.fit_predict(embeddings)
    return labels


def cluster_kmeans(embeddings: np.ndarray, n_clusters: int = 20) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=0)
    labels = km.fit_predict(embeddings)
    return labels


def top_examples_per_cluster(df: pd.DataFrame, labels: np.ndarray, top_n: int = 5) -> pd.DataFrame:
    df2 = df.copy()
    df2['cluster'] = labels
    out_rows = []
    for label, group in df2.groupby('cluster'):
        size = len(group)
        examples = group['question'].head(top_n).tolist()
        out_rows.append({'cluster': int(label), 'size': int(size), 'examples': examples})
    out = pd.DataFrame(out_rows).sort_values(by='size', ascending=False).reset_index(drop=True)
    return out


def plot_2d(emb2d: np.ndarray, labels: np.ndarray, out_path: str):
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    palette = plt.cm.get_cmap('tab20', len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        alpha = 0.6 if lab != -1 else 0.3
        size = 8 if lab != -1 else 5
        plt.scatter(emb2d[mask, 0], emb2d[mask, 1], s=size, label=str(lab), alpha=alpha)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize='small')
    plt.title('2D projection of question clusters')
    plt.xlabel('dim1')
    plt.ylabel('dim2')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    df = load_questions(args.input, text_col=args.text_col)
    texts = df['question'].tolist()
    print(f"Loaded {len(texts)} unique questions")

    # Embed
    if args.backend == 'st':
        model_name = args.model or 'all-MiniLM-L6-v2'
        embeddings = embed_with_st(model_name, texts, batch_size=args.batch_size)
    elif args.backend == 'openai':
        if not args.openai_key:
            raise RuntimeError('OpenAI backend selected but --openai_key not provided')
        model_name = args.model or 'text-embedding-3-small'
        embeddings = embed_with_openai(args.openai_key, model_name, texts, batch_size=args.batch_size)
    else:
        raise ValueError('Unknown backend: choose "st" or "openai"')

    print('Embeddings shape:', embeddings.shape)

    # Optional reduce dims for clustering
    if args.use_umap:
        emb_for_clustering = reduce_dim(embeddings, n_components=args.umap_dims, random_state=args.random_seed)
        print('Reduced embeddings shape for clustering:', emb_for_clustering.shape)
    else:
        emb_for_clustering = embeddings

    # Choose clustering algorithm
    if args.cluster_algo == 'hdbscan':
        labels = cluster_hdbscan(emb_for_clustering, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples)
    else:
        # KMeans
        n_clusters = args.n_clusters or 20
        labels = cluster_kmeans(emb_for_clustering, n_clusters=n_clusters)

    df['cluster'] = labels

    # Optional 2D projection for plotting
    if args.plot:
        try:
            emb2d = reduce_dim(embeddings, n_components=2, random_state=args.random_seed)
            plot_2d(emb2d, labels, args.plot_output)
            print('Saved cluster plot to', args.plot_output)
        except Exception as e:
            print('Failed to make 2D plot:', e)

    # Save outputs
    out_csv = args.output or 'clustered_questions.csv'
    df.to_csv(out_csv, index=False)
    print('Saved clustered questions to', out_csv)

    # Show cluster summary
    summary = top_examples_per_cluster(df[['question']], labels, top_n=args.top_examples)
    summary_csv = args.summary_output or 'cluster_summary.csv'
    summary.to_csv(summary_csv, index=False)
    print('Saved cluster summary to', summary_csv)

    # If using KMeans, compute silhouette
    if args.cluster_algo == 'kmeans':
        try:
            score = silhouette_score(emb_for_clustering, labels)
            print(f'Silhouette score for KMeans: {score:.4f}')
        except Exception:
            pass

    print('Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster natural language questions by semantic meaning')
    parser.add_argument('--input', required=True, help='Path to questions file (CSV or TXT)')
    parser.add_argument('--text_col', default=None, help='Column name in CSV that contains the question text')
    parser.add_argument('--backend', choices=['st', 'openai'], default='st', help='Embedding backend: st (sentence-transformers) or openai')
    parser.add_argument('--model', default=None, help='Embedding model name (depends on backend)')
    parser.add_argument('--openai_key', default=None, help='OpenAI API key (if using openai backend)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_umap', action='store_true', help='Use UMAP to reduce dims before clustering (recommended)')
    parser.add_argument('--umap_dims', type=int, default=15, help='UMAP dims used for clustering')
    parser.add_argument('--random_seed', type=int, default=42)

    parser.add_argument('--cluster_algo', choices=['hdbscan', 'kmeans'], default='kmeans')
    parser.add_argument('--min_cluster_size', type=int, default=10)
    parser.add_argument('--min_samples', type=int, default=None)
    parser.add_argument('--n_clusters', type=int, default=20)

    parser.add_argument('--plot', action='store_true', help='Create a 2D scatter plot of the clusters')
    parser.add_argument('--plot_output', default='clusters.png', help='Path to save 2D cluster plot')
    parser.add_argument('--output', default='clustered_questions.csv', help='CSV output with cluster labels')
    parser.add_argument('--summary_output', default='cluster_summary.csv', help='CSV output with cluster sizes and examples')
    parser.add_argument('--top_examples', type=int, default=5)

    args = parser.parse_args()
    main(args)
