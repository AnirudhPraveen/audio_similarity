from audio_similarity import AudioSimilaritySearch, IndexType
from pathlib import Path

# Initialize
searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)

# Index your files
dataset_dir = Path("~/Documents/data").expanduser()
audio_files = list(dataset_dir.glob("**/*.wav"))
searcher.add_batch(audio_files)

# Search
query_file = Path("~/Documents/RogerMoore_2.wav").expanduser()
results = searcher.search(str(query_file), k=5)

# Visualize search results
searcher.visualize_search_results(
    query_path=str(query_file),
    results=results,
    save_path="search_results.png",
    show=True  # Set to False to save without displaying
)

# Run and visualize benchmark
configs = [
    {'type': IndexType.FLAT},
    {'type': IndexType.IVF, 'params': {'nlist': 100}},
    {'type': IndexType.HNSW, 'params': {'M': 16}}
]

results = searcher.benchmark(
    compare_with=configs,
    num_samples=1000,
    num_queries=100,
    k=5
)

# Visualize all metrics
searcher.visualize_benchmarks(
    metric="all",
    save_path="benchmark_all.png",
    show=True
)

# Or visualize specific metrics
for metric in ["build_time", "query_time", "memory_usage", "recall"]:
    searcher.visualize_benchmarks(
        metric=metric,
        save_path=f"benchmark_{metric}.png",
        show=True
    )
# """Basic example of using the audio similarity library with index saving."""

# from pathlib import Path
# from audio_similarity import AudioSimilaritySearch, IndexType
# import time

# def main():
#     # Initialize
#     searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
    
#     # Set up paths
#     dataset_dir = Path("~/Documents/data").expanduser()
#     query_file = Path("~/Documents/RogerMoore_2.wav").expanduser()
#     save_dir = Path("saved_indices")  # Directory to save indices
#     save_dir.mkdir(exist_ok=True)
    
#     print(f"Using dataset from: {dataset_dir}")
    
#     # Get all wav files
#     audio_files = list(dataset_dir.glob("**/*.wav"))
#     print(f"Found {len(audio_files)} audio files")
    
#     # Index the files
#     print("Indexing files...")
#     searcher.add_batch(audio_files)
    
#     # Save the index with timestamp
#     timestamp = time.strftime("%Y%m%d-%H%M%S")
#     index_path = save_dir / f"index_{timestamp}"
#     print(f"\nSaving index to: {index_path}")
#     searcher.save(index_path)
#     print("Index saved successfully")
    
#     # Load the index (demonstrating how to load)
#     print("\nLoading saved index...")
#     new_searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
#     new_searcher.load(index_path)
#     print("Index loaded successfully")
    
#     # Search using loaded index
#     print(f"\nSearching for files similar to: {query_file.name}")
#     results = new_searcher.search(str(query_file), k=5)
    
#     # Print results
#     print("\nSearch Results:")
#     print("-" * 50)
#     for i, (file_path, distance) in enumerate(results, 1):
#         print(f"{i}. File: {Path(file_path).name}")
#         print(f"   Distance: {distance:.4f}")

# if __name__ == "__main__":
#     main()