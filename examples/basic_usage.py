"""Basic example with debug prints."""

import numpy as np
import torch
import torchaudio
from pathlib import Path
from audio_similarity import AudioSimilaritySearch, IndexType

def main():
    """Run basic example with debug information."""
    print("Starting basic example...")
    
    # Create test directory
    data_dir = Path("example_data")
    data_dir.mkdir(exist_ok=True)
    audio_dir = data_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    
    print(f"\nUsing directories:")
    print(f"Data directory: {data_dir}")
    print(f"Audio directory: {audio_dir}")
    
    # Generate test audio
    print("\nGenerating test audio files...")
    sample_rate = 16000
    duration = 1  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    
    audio_files = []
    for freq in [440, 880, 1320]:  # A4, A5, E6
        waveform = torch.sin(2 * np.pi * freq * t).unsqueeze(0)
        file_path = audio_dir / f"tone_{freq}hz.wav"
        torchaudio.save(str(file_path), waveform, sample_rate)
        audio_files.append(file_path)
        print(f"Generated: {file_path}")
    
    # Initialize search
    print("\nInitializing audio similarity search...")
    searcher = AudioSimilaritySearch(index_type=IndexType.FLAT)
    
    # Index files
    print("\nIndexing audio files...")
    indices = searcher.add_batch(audio_files)
    print(f"Indexed {len(indices)} files")
    
    # Search
    print("\nPerforming search...")
    query_file = audio_files[0]
    results = searcher.search(query_file, k=2)
    
    print("\nSearch Results:")
    for i, (file_path, distance) in enumerate(results, 1):
        print(f"{i}. File: {Path(file_path).name}")
        print(f"   Distance: {distance:.4f}")
    
    # Visualize
    print("\nGenerating visualization...")
    searcher.visualize_search_results(
        query_file,
        results,
        save_path=data_dir / "results.png"
    )
    print(f"Visualization saved to: {data_dir/'results.png'}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()