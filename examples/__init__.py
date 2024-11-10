"""Example scripts for audio similarity search library."""

from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent
DATA_DIR = EXAMPLES_DIR / "data"
MODELS_DIR = EXAMPLES_DIR / "saved_models"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)