from audio_similarity import AudioSimilaritySearch, IndexType
from pathlib import Path

def main():
    # Initialize
    searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
    
    # Set up dataset
    dataset_dir = Path("~/Documents/data").expanduser()
    query_file = Path("~/Documents/RogerMoore_2.wav").expanduser()
    
    # Get audio files
    audio_files = list(dataset_dir.glob("**/*.wav"))
    print(f"Found {len(audio_files)} audio files")
    
    # Index files
    #searcher.add_batch(audio_files)

    saved_index_dir = Path("./saved_indices/index_20241111-094634").expanduser()

    searcher.load(saved_index_dir)
    
    # 1. Get Search Results
    print("\n1. Search Results:")
    print("-" * 50)
    results = searcher.search(str(query_file), k=5)
    for i, (file_path, distance) in enumerate(results, 1):
        print(f"{i}. File: {Path(file_path).name}")
        print(f"   Distance: {distance:.4f}")
    
    # 2. Visualize Search Results
    searcher.visualize_search_results(
        query_path=str(query_file),
        results=results,
        save_path="search_results.png",
        show=True
    )

    print(results)
    
    # # 3. Run Benchmarks
    # print("\n3. Running Benchmarks...")
    # configs = [
    #     {'type': IndexType.FLAT},
    #     {'type': IndexType.IVF, 'params': {'nlist': 100}},
    #     {'type': IndexType.HNSW, 'params': {'M': 16}},
    #     {'type': IndexType.PQ, 'params': {'M': 8, 'nbits': 8}}
    # ]
    
    # benchmark_results = searcher.benchmark(
    #     compare_with=configs,
    #     num_samples=min(1000, len(audio_files)),
    #     num_queries=min(100, len(audio_files) // 10),
    #     k=5
    # )
    
    # # 4. Print Benchmark Results
    # print("\nBenchmark Results:")
    # print("-" * 50)
    # for result in benchmark_results:
    #     print(f"\nIndex Type: {result.index_type}")
    #     print(f"Build Time: {result.build_time:.4f}s")
    #     print(f"Query Time: {result.query_time:.6f}s")
    #     print(f"Memory Usage: {result.memory_usage:.2f}MB")
    #     print(f"Recall@{result.k}: {result.recall:.4f}")
    #     if result.params:
    #         print("Parameters:", result.params)
    
    # # 5. Save and show all visualizations
    # # Make sure benchmark results are stored
    # searcher.benchmark_results = benchmark_results
    
    # # Overall benchmark visualization
    # searcher.visualize_benchmarks(
    #     metric="all",
    #     save_path="benchmark_all.png",
    #     show=True
    # )
    
    # # Individual metric visualizations
    # for metric in ["build_time", "query_time", "memory_usage", "recall"]:
    #     searcher.visualize_benchmarks(
    #         metric=metric,
    #         save_path=f"benchmark_{metric}.png",
    #         show=True
    #     )

if __name__ == "__main__":
    main()