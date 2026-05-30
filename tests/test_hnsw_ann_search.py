"""
Tests for HNSW approximate nearest neighbor search in ContentRecommender (Issue #315).
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics.pairwise import cosine_similarity

try:
    import hnswlib  # noqa: F401
    _HNSWLIB_INSTALLED = True
except ImportError:
    _HNSWLIB_INSTALLED = False

requires_hnswlib = pytest.mark.skipif(
    not _HNSWLIB_INSTALLED,
    reason="hnswlib not installed — skipping HNSW index tests",
)

from src.model.content_model import ContentRecommender


@pytest.fixture
def sample_item_df():
    return pd.DataFrame({
        'title': [
            'Harry Potter',
            'Lord of the Rings',
            'The Hobbit',
            'Game of Thrones',
            'Dune',
        ],
        'description': [
            'A young wizard discovers his magical heritage',
            'A fellowship embarks on a quest to destroy a ring',
            'A hobbit goes on an unexpected journey',
            'Noble families fight for control of the Iron Throne',
            'A desert planet holds the most valuable resource',
        ],
        'category': ['Fantasy', 'Fantasy', 'Fantasy', 'Fantasy', 'SciFi'],
        'combined': [
            'Harry Potter A young wizard discovers his magical heritage Fantasy',
            'Lord of the Rings A fellowship embarks on a quest Fantasy',
            'The Hobbit A hobbit goes on an unexpected journey Fantasy',
            'Game of Thrones Noble families fight for the Iron Throne Fantasy',
            'Dune A desert planet holds the most valuable resource SciFi',
        ],
    })


@pytest.fixture
def embedding_matrix():
    """Fixed unit vectors so HNSW and brute-force paths are comparable without loading ST models."""
    rng = np.random.RandomState(42)
    raw = rng.randn(5, 16).astype(np.float32)
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


@pytest.fixture
def fake_sentence_transformer(embedding_matrix, sample_item_df):
    class _FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts, show_progress_bar=False):
            if isinstance(texts, list) and len(texts) == len(sample_item_df):
                return embedding_matrix.copy()
            query = texts[0] if isinstance(texts, list) else texts
            if 'wizard' in str(query).lower():
                return embedding_matrix[0:1].copy()
            return embedding_matrix[1:2].copy()

    return _FakeSentenceTransformer


@pytest.fixture
def content_model(sample_item_df, fake_sentence_transformer, monkeypatch):
    monkeypatch.setattr(
        'src.model.content_model.SentenceTransformer',
        fake_sentence_transformer,
    )
    return ContentRecommender(sample_item_df)


@requires_hnswlib
class TestHNSWIndex:
    def test_hnsw_index_built(self, content_model):
        assert content_model._hnsw_index is not None
        assert content_model._hnsw_index.element_count == len(content_model.df)

    def test_recommend_uses_hnsw_and_matches_brute_force_top_titles(self, content_model):
        title = 'Harry Potter'
        top_n = 3

        hnsw_recs = content_model.recommend(title, top_n=top_n)
        assert len(hnsw_recs) == top_n

        idx = content_model._title_to_idx[title.lower()]
        query_vec = content_model.matrix[idx].reshape(1, -1)
        scores = cosine_similarity(query_vec, content_model.matrix).flatten()
        brute_order = sorted(
            ((i, float(scores[i])) for i in range(len(scores)) if i != idx),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        brute_titles = [content_model.df.iloc[i]['title'] for i, _ in brute_order]

        hnsw_titles = [r['title'] for r in hnsw_recs]
        assert hnsw_titles == brute_titles

    def test_search_returns_results_with_hnsw(self, content_model):
        results = content_model.search('wizard', top_n=3)
        assert len(results) >= 1
        assert all('title' in r and 'score' in r for r in results)


class TestHNSWFallback:
    def test_fallback_when_hnsw_unavailable(
        self, sample_item_df, embedding_matrix, fake_sentence_transformer, monkeypatch,
    ):
        import src.model.content_model as cm

        monkeypatch.setattr(cm, '_HNSWLIB_AVAILABLE', False)
        monkeypatch.setattr(cm, 'SentenceTransformer', fake_sentence_transformer)

        model = ContentRecommender(sample_item_df)
        assert model._hnsw_index is None
        recs = model.recommend('Harry Potter', top_n=2)
        assert len(recs) == 2
