"""Tests for the index_factory module."""

import pytest
import numpy as np
import faiss

from audio_similarity.index_factory import IndexFactory, IndexType

class TestIndexFactory:
    """Test cases for IndexFactory class."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.dimension = 768  # wav2vec2 dimension
        self.num_samples = 1000
        self.data = np.random.random((self.num_samples, self.dimension)).astype('float32')

    def test_create_flat_index(self):
        """Test creation of flat index."""
        index = IndexFactory.create_index(self.dimension, IndexType.FLAT)
        
        assert isinstance(index, faiss.IndexFlatL2)
        assert index.d == self.dimension
        
        # Test search functionality
        index.add(self.data)
        query = np.random.random((1, self.dimension)).astype('float32')
        D, I = index.search(query, k=5)
        
        assert D.shape == (1, 5)
        assert I.shape == (1, 5)

    def test_create_ivf_index(self):
        """Test creation of IVF index."""
        nlist = 100
        index = IndexFactory.create_index(
            self.dimension,
            IndexType.IVF,
            nlist=nlist
        )
        
        assert isinstance(index, faiss.IndexIVFFlat)
        assert index.nlist == nlist
        
        # Test training and search
        index.train(self.data)
        index.add(self.data)
        query = np.random.random((1, self.dimension)).astype('float32')
        D, I = index.search(query, k=5)
        
        assert D.shape == (1, 5)
        assert I.shape == (1, 5)

    def test_create_hnsw_index(self):
        """Test creation of HNSW index."""
        M = 16
        index = IndexFactory.create_index(
            self.dimension,
            IndexType.HNSW,
            M=M
        )
        
        assert isinstance(index, faiss.IndexHNSWFlat)
        assert index.hnsw.M == M
        
        # Test search functionality
        index.add(self.data)
        query = np.random.random((1, self.dimension)).astype('float32')
        D, I = index.search(query, k=5)
        
        assert D.shape == (1, 5)
        assert I.shape == (1, 5)

    def test_create_pq_index(self):
        """Test creation of PQ index."""
        M = 8
        nbits = 8
        index = IndexFactory.create_index(
            self.dimension,
            IndexType.PQ,
            M=M,
            nbits=nbits
        )
        
        assert isinstance(index, faiss.IndexPQ)
        
        # Test training and search
        index.train(self.data)
        index.add(self.data)
        query = np.random.random((1, self.dimension)).astype('float32')
        D, I = index.search(query, k=5)
        
        assert D.shape == (1, 5)
        assert I.shape == (1, 5)

    def test_invalid_index_type(self):
        """Test creation with invalid index type."""
        with pytest.raises(ValueError):
            IndexFactory.create_index(self.dimension, "invalid_type")

    def test_index_parameters(self):
        """Test index parameter validation."""
        # Test HNSW parameters
        index = IndexFactory.create_index(
            self.dimension,
            IndexType.HNSW,
            M=16,
            efConstruction=40,
            efSearch=16
        )
        assert index.hnsw.efConstruction == 40
        assert index.hnsw.efSearch == 16
        
        # Test IVF parameters
        index = IndexFactory.create_index(
            self.dimension,
            IndexType.IVF,
            nlist=100
        )
        assert index.nlist == 100
        
        # Test PQ parameters
        index = IndexFactory.create_index(
            self.dimension,
            IndexType.PQ,
            M=8,
            nbits=8
        )
        assert index.M == 8

    def test_search_accuracy(self):
        """Test search accuracy of different indices."""
        # Create ground truth with flat index
        flat_index = IndexFactory.create_index(self.dimension, IndexType.FLAT)
        flat_index.add(self.data)
        query = np.random.random((10, self.dimension)).astype('float32')
        ground_truth, _ = flat_index.search(query, k=5)
        
        # Test other indices
        for idx_type in [IndexType.IVF, IndexType.HNSW, IndexType.PQ]:
            index = IndexFactory.create_index(self.dimension, idx_type)
            if hasattr(index, 'train'):
                index.train(self.data)
            index.add(self.data)
            results, _ = index.search(query, k=5)
            
            # Check if results are reasonably close to ground truth
            assert np.abs(results - ground_truth).mean() < 1.0