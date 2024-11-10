"""
FAISS index creation and management.

This module provides functionality for creating and managing different types
of FAISS indices for similarity search.
"""

from enum import Enum
from typing import Optional, Dict, Any
import logging

import numpy as np
import faiss

logger = logging.getLogger(__name__)

class IndexType(Enum):
    """
    Enumeration of supported FAISS index types.

    Attributes
    ----------
    FLAT : str
        Exact search using L2 distance
    IVF : str
        Inverted file index with approximate search
    HNSW : str
        Hierarchical Navigable Small World graph-based index
    PQ : str
        Product Quantization-based index
    """
    
    FLAT = "Flat"
    IVF = "IVF"
    HNSW = "HNSW"
    PQ = "PQ"

class IndexFactory:
    """
    Factory class for creating FAISS indices.

    This class provides methods to create and configure different types of
    FAISS indices based on the desired search characteristics.

    Notes
    -----
    The choice of index type affects the trade-off between:
    - Search accuracy
    - Search speed
    - Memory usage
    - Build time
    """

    @staticmethod
    def create_index(
        dimension: int,
        index_type: IndexType,
        **kwargs: Any
    ) -> faiss.Index:
        """
        Create a FAISS index of the specified type.

        Parameters
        ----------
        dimension : int
            Dimensionality of the vectors
        index_type : IndexType
            Type of index to create
        **kwargs : Dict[str, Any]
            Additional parameters for specific index types

        Returns
        -------
        faiss.Index
            Configured FAISS index

        Raises
        ------
        ValueError
            If the index type is unknown or parameters are invalid

        Examples
        --------
        >>> # Create a flat index
        >>> index = IndexFactory.create_index(128, IndexType.FLAT)
        
        >>> # Create an IVF index with custom parameters
        >>> index = IndexFactory.create_index(128, IndexType.IVF, nlist=100)
        """
        if index_type == IndexType.FLAT:
            return IndexFactory._create_flat_index(dimension)
            
        elif index_type == IndexType.IVF:
            return IndexFactory._create_ivf_index(dimension, **kwargs)
            
        elif index_type == IndexType.HNSW:
            return IndexFactory._create_hnsw_index(dimension, **kwargs)
            
        elif index_type == IndexType.PQ:
            return IndexFactory._create_pq_index(dimension, **kwargs)
            
        raise ValueError(f"Unknown index type: {index_type}")

    @staticmethod
    def _create_flat_index(dimension: int) -> faiss.IndexFlatL2:
        """
        Create a flat index for exact search.

        Parameters
        ----------
        dimension : int
            Dimensionality of the vectors

        Returns
        -------
        faiss.IndexFlatL2
            Flat index using L2 distance

        Notes
        -----
        Flat index provides exact search but scales linearly with dataset size
        """
        return faiss.IndexFlatL2(dimension)

    @staticmethod
    def _create_ivf_index(
        dimension: int,
        nlist: int = 100,
        **kwargs: Any
    ) -> faiss.IndexIVFFlat:
        """
        Create an IVF index for approximate search.

        Parameters
        ----------
        dimension : int
            Dimensionality of the vectors
        nlist : int, optional
            Number of centroids, by default 100
        **kwargs : Any
            Additional parameters

        Returns
        -------
        faiss.IndexIVFFlat
            IVF index configured with given parameters

        Notes
        -----
        IVF index requires training before use
        """
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train_required = True
        return index

    @staticmethod
    def _create_hnsw_index(
        dimension: int,
        M: int = 16,
        efConstruction: int = 40,
        efSearch: int = 16,
        **kwargs: Any
    ) -> faiss.IndexHNSWFlat:
        """
        Create an HNSW index for approximate search.

        Parameters
        ----------
        dimension : int
            Dimensionality of the vectors
        M : int, optional
            Number of connections per layer, by default 16
        efConstruction : int, optional
            Size of the dynamic candidate list for construction, by default 40
        efSearch : int, optional
            Size of the dynamic candidate list for search, by default 16
        **kwargs : Any
            Additional parameters

        Returns
        -------
        faiss.IndexHNSWFlat
            HNSW index configured with given parameters

        Notes
        -----
        HNSW provides good speed/accuracy trade-off but uses more memory
        """
        index = faiss.IndexHNSWFlat(dimension, M)
        index.hnsw.efConstruction = efConstruction
        index.hnsw.efSearch = efSearch
        return index

    @staticmethod
    def _create_pq_index(
        dimension: int,
        M: int = 8,
        nbits: int = 8,
        **kwargs: Any
    ) -> faiss.IndexPQ:
        """
        Create a Product Quantization index for compact storage.

        Parameters
        ----------
        dimension : int
            Dimensionality of the vectors
        M : int, optional
            Number of subquantizers, by default 8
        nbits : int, optional
            Number of bits per subquantizer, by default 8
        **kwargs : Any
            Additional parameters

        Returns
        -------
        faiss.IndexPQ
            PQ index configured with given parameters

        Notes
        -----
        PQ provides compact storage but lower accuracy
        """
        index = faiss.IndexPQ(dimension, M, nbits)
        index.train_required = True
        return index