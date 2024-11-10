from pathlib import Path
from audio_similarity import AudioSimilaritySearch, IndexType

# Initialize with index persistence
searcher = AudioSimilaritySearch(
    index_type=IndexType.FLAT,
    index_path="saved_index.faiss"
)

# Get all audio files in directory
audio_dir = Path("path/to/audio/files")
audio_files = list(audio_dir.glob("*.wav"))

# Add files in batch if index doesn't exist
if not searcher.is_indexed():
    print(f"Building index for {len(audio_files)} files...")
    searcher.add_batch(audio_files)
    print("Index built and saved!")
else:
    print("Using existing index")

# Search
query_path = "path/to/query.wav"
results = searcher.search(query_path, k=5)

# Print results
print("\nSearch Results:")
for file_path, distance in results:
    print(f"Similar file: {file_path}, distance: {distance:.4f}")

# Visualize
searcher.visualize_search_results(
    query_path,
    results,
    save_path="results.png"
)