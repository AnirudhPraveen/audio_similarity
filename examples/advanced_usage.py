import numpy as np
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import hashlib
import time 
import json

from audio_similarity import AudioSimilaritySearch, IndexType

# def setup_argparse():
#     """Set up command line argument parsing."""
#     parser = argparse.ArgumentParser(description="Audio similarity search with custom dataset location")
#     parser.add_argument(
#         "--dataset_path",
#         type=str,
#         required=True,
#         help="Path to the directory containing audio files"
#     )
#     parser.add_argument(
#         "--output_dir",
#         type=str,
#         default="./output",
#         help="Directory for storing output files and visualizations"
#     )
#     parser.add_argument(
#         "--query",
#         type=str,
#         help="Path to the query audio file"
#     )
#     parser.add_argument(
#         "--top_k",
#         type=int,
#         default=5,
#         help="Number of similar results to return"
#     )
#     parser.add_argument(
#         "--benchmark",
#         action="store_true",
#         help="Run benchmark comparisons of different index types"
#     )
#     return parser

def setup_argparse():
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(description="Audio similarity search with custom dataset location")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the directory containing audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory for storing output files and visualizations"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Path to the query audio file"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of similar results to return"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparisons of different index types"
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild index even if it exists"
    )
    return parser

def get_dataset_hash(audio_files):
    """Generate a hash of the dataset based on file paths and modification times."""
    content = []
    for file_path in sorted(audio_files):
        mtime = Path(file_path).stat().st_mtime
        content.append(f"{file_path}:{mtime}")
    
    return hashlib.md5("|".join(content).encode()).hexdigest()

def get_index_path(output_dir: Path, dataset_hash: str, index_config: dict):
    """Generate paths for index and metadata files."""
    config_str = f"{index_config['type'].name}"
    if 'params' in index_config:
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(index_config['params'].items()))
        config_str += f"_{param_str}"
    
    base_name = f"index_{dataset_hash}_{config_str}"
    return {
        'index': output_dir / f"{base_name}.faiss",
        'metadata': output_dir / f"{base_name}.json"
    }

def save_index(searcher, audio_files, index_config: dict, output_dir: Path):
    """Save the index and metadata."""
    dataset_hash = get_dataset_hash(audio_files)
    paths = get_index_path(output_dir, dataset_hash, index_config)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the index
    searcher.save_index(str(paths['index']))
    
    # Save metadata
    metadata = {
        'dataset_hash': dataset_hash,
        'index_config': {
            'type': index_config['type'].name,
            'params': index_config.get('params', {})
        },
        'creation_time': time.time(),
        'num_files': len(audio_files),
        'file_paths': [str(f) for f in audio_files]
    }
    
    with open(paths['metadata'], 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return paths

def load_index(audio_files, index_config: dict, output_dir: Path):
    """Try to load existing index if it matches the current dataset."""
    dataset_hash = get_dataset_hash(audio_files)
    paths = get_index_path(output_dir, dataset_hash, index_config)
    
    # Check if both index and metadata exist
    if not (paths['index'].exists() and paths['metadata'].exists()):
        return None
    
    # Load and verify metadata
    try:
        with open(paths['metadata'], 'r') as f:
            metadata = json.load(f)
        
        # Verify dataset hash
        if metadata['dataset_hash'] != dataset_hash:
            return None
        
        # Initialize searcher with saved configuration
        searcher = AudioSimilaritySearch(
            index_type=IndexType[metadata['index_config']['type']],
            index_params=metadata['index_config']['params']
        )
        
        # Load the index
        searcher.load_index(str(paths['index']))
        
        print(f"Loaded existing index from {paths['index']}")
        return searcher
    
    except Exception as e:
        print(f"Error loading index: {e}")
        return None

def get_optimal_index_config(num_vectors: int):
    """
    Get optimal index configuration based on dataset size.
    
    Args:
        num_vectors: Number of vectors in the dataset
    
    Returns:
        dict: Recommended index configuration
    """
    if num_vectors < 10000:
        print("Small dataset detected: Using FLAT index for exact search")
        return {'type': IndexType.FLAT}
    
    elif num_vectors < 50000:
        # For medium datasets, use IVF
        nlist = int(np.sqrt(num_vectors))
        print(f"Medium dataset detected: Using IVF index with {nlist} clusters")
        return {
            'type': IndexType.IVF,
            'params': {'nlist': nlist}
        }
    
    else:
        # For large datasets, use IVFPQ
        nlist = int(np.sqrt(num_vectors))
        # Adjust M based on dataset size
        m = 8 if num_vectors < 100000 else 16
        print(f"Large dataset detected: Using IVFPQ index with {nlist} clusters and {m} subquantizers")
        return {
            'type': IndexType.PQ,
            'params': {
                'M': m,
                'nbits': 8,
                'nlist': nlist
            }
        }

def get_benchmark_configs(num_vectors: int):
    """
    Get benchmark configurations based on dataset size.
    
    Args:
        num_vectors: Number of vectors in the dataset
    
    Returns:
        list: List of index configurations to benchmark
    """
    configs = [{'type': IndexType.FLAT}]  # Always include FLAT as baseline
    
    # Calculate optimal parameters
    nlist = int(np.sqrt(num_vectors))
    
    if num_vectors >= 10000:
        configs.append({
            'type': IndexType.IVF,
            'params': {'nlist': nlist}
        })
    
    if num_vectors >= 50000:
        configs.append({
            'type': IndexType.PQ,
            'params': {
                'M': 8,
                'nbits': 8,
                'nlist': nlist
            }
        })
        
        configs.append({
            'type': IndexType.HNSW,
            'params': {'M': 16 if num_vectors < 1000000 else 32}
        })
    
    return configs

def load_audio_files(dataset_path: Path):
    """Load audio files from the specified directory."""
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg'}
    audio_files = []
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
    
    for file_path in dataset_path.rglob('*'):
        if file_path.suffix.lower() in supported_formats:
            audio_files.append(file_path)
    
    if not audio_files:
        raise ValueError(f"No audio files found in {dataset_path}. Please provide a directory containing audio files.")
    
    return audio_files

def verify_query_file(query_path: str):
    """Verify that the query file exists and is a supported audio format."""
    supported_formats = {'.wav', '.mp3', '.flac', '.ogg'}
    query_path = Path(query_path)
    
    if not query_path.exists():
        raise FileNotFoundError(f"Query file {query_path} does not exist")
    
    if query_path.suffix.lower() not in supported_formats:
        raise ValueError(f"Query file format {query_path.suffix} not supported. Supported formats: {supported_formats}")
    
    return query_path

def compare_indices(audio_files, output_dir: Path):
    """Compare different FAISS index types."""
    num_vectors = len(audio_files)
    configs = get_benchmark_configs(num_vectors)
    
    # Adjust benchmark parameters based on dataset size
    num_samples = min(1000, num_vectors)
    num_queries = min(10, num_vectors // 10)
    
    searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
    
    print("Indexing files for benchmark...")
    searcher.add_batch(audio_files)
    
    print("\nBenchmarking different index types...")
    print(f"Using {num_samples} samples and {num_queries} queries")
    results = searcher.benchmark(
        compare_with=configs,
        num_samples=num_samples,
        num_queries=num_queries,
        k=5
    )
    
    output_dir.mkdir(exist_ok=True, parents=True)
    searcher.visualize_benchmarks(
        metric="all",
        save_path=output_dir / "benchmark_results.png"
    )
    
    return results

# def search_similar_audio(audio_files, query_file: Path, top_k: int = 5):
#     """Search for similar audio files to the query."""
#     num_vectors = len(audio_files)
#     config = get_optimal_index_config(num_vectors)
    
#     searcher = AudioSimilaritySearch(
#         index_type=config['type'],
#         index_params=config.get('params', {})
#     )
    
#     # Index all files
#     print("Indexing audio files...")
#     searcher.add_batch(audio_files)
    
#     # Perform search
#     print(f"\nSearching for top {top_k} matches to {query_file.name}...")
#     results = searcher.search(query_file, k=top_k)
    
#     return results

def search_similar_audio(audio_files, query_file: Path, output_dir: Path, top_k: int = 5, force_rebuild: bool = False):
    """Search for similar audio files to the query."""
    num_vectors = len(audio_files)
    config = get_optimal_index_config(num_vectors)
    
    # Try to load existing index
    if not force_rebuild:
        searcher = load_index(audio_files, config, output_dir)
        if searcher is not None:
            print("Using existing index...")
        else:
            print("No matching index found. Building new index...")
    else:
        searcher = None
    
    # Build new index if needed
    if searcher is None:
        searcher = AudioSimilaritySearch(
            index_type=config['type'],
            index_params=config.get('params', {})
        )
        
        print("Indexing audio files...")
        searcher.add_batch(audio_files)
        
        # Save the index for future use
        print("Saving index for future use...")
        paths = save_index(searcher, audio_files, config, output_dir)
        print(f"Index saved to {paths['index']}")
    
    # Perform search
    print(f"\nSearching for top {top_k} matches to {query_file.name}...")
    results = searcher.search(query_file, k=top_k)
    
    return results

def main():
    """Run advanced example with custom dataset location and query."""
    # Parse command line arguments
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Convert string paths to Path objects
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    
    print(f"Using dataset from: {dataset_path}")
    print(f"Outputs will be saved to: {output_dir}")
    
    # Load dataset
    print("\n1. Loading audio files...")
    try:
        audio_files = load_audio_files(dataset_path)
        num_files = len(audio_files)
        print(f"Found {num_files} audio files")
        
        # Print recommended configuration based on dataset size
        optimal_config = get_optimal_index_config(num_files)
        print("\nRecommended index configuration:")
        print(f"Index Type: {optimal_config['type']}")
        if 'params' in optimal_config:
            print("Parameters:")
            for key, value in optimal_config['params'].items():
                print(f"  {key}: {value}")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        return
    
    # Run benchmark if requested
    if args.benchmark:
        print("\n2. Running index type comparison benchmark...")
        benchmark_results = compare_indices(audio_files, output_dir)
        
        print("\nBenchmark Results:")
        print("-" * 50)
        for result in benchmark_results:
            print(f"\nIndex Type: {result.index_type}")
            print(f"Build Time: {result.build_time:.4f}s")
            print(f"Query Time: {result.query_time:.6f}s")
            print(f"Memory Usage: {result.memory_usage:.2f}MB")
            print(f"Recall@{result.k}: {result.recall:.4f}")
    
    # Process query if provided
    if args.query:
        try:
            query_file = verify_query_file(args.query)
            results = search_similar_audio(audio_files, query_file, args.top_k)
            
            print("\nSearch Results:")
            print("-" * 50)
            for i, (file_path, distance) in enumerate(results, 1):
                print(f"{i}. File: {Path(file_path).name}")
                print(f"   Distance: {distance:.4f}")
                
        except (FileNotFoundError, ValueError) as e:
            print(f"Error with query file: {e}")
            return

if __name__ == "__main__":
    main()