import pathlib
import os


CURRENT_FILE = pathlib.Path(__file__).resolve()

PROJECT_ROOT = CURRENT_FILE.parent.parent.parent

OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_OUTPUT_DIR = OUTPUT_DIR / "model"
LOG_OUTPUT_DIR = OUTPUT_DIR / "log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === DATASET CONFIG ===
DATASET_DIR = PROJECT_ROOT / "dataset"

MIND_DATASET_DIR = DATASET_DIR / "mind"

CACHE_DIR = PROJECT_ROOT / ".cache"

# MIND Small
MIND_SMALL_DATASET_DIR = MIND_DATASET_DIR / "small"
MIND_SMALL_VAL_DATASET_DIR = pathlib.Path("/content/data/small_dev")
MIND_SMALL_TRAIN_DATASET_DIR = pathlib.Path("/content/data/small_train")

# MIND Large (gội cả train và valid lại để train)
MIND_LARGE_TRAIN_DATASET_DIR = pathlib.Path("/content/data/large_full")








