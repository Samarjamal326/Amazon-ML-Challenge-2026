"""Paths and global settings.

Override with environment variables:
  ER_DATA_DIR   -- directory containing train/ and test/ subdirs with TSV files
  ER_CACHE_DIR  -- scratch directory for intermediate parquet/model files
  ER_OUTPUT_DIR -- where matching_results.tsv and candidate_pairs.tsv are written
"""
import os
from pathlib import Path

# Resolve the actual dataset location under the student_resource bundle
_WORKSPACE = Path(__file__).resolve().parents[1]
_DEFAULT_DATA = (
    _WORKSPACE
    / "dataset"
    / "6ab10eb3b23ba_student_resource"
    / "student_resource"
    / "dataset"
)

DATA_DIR = Path(os.environ.get("ER_DATA_DIR", _DEFAULT_DATA))
CACHE_DIR = Path(os.environ.get("ER_CACHE_DIR", _WORKSPACE / "pipeline_cache"))
OUTPUT_DIR = Path(os.environ.get("ER_OUTPUT_DIR", _WORKSPACE / "outputs"))

SEED = 42

for _d in (CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def raw_path(split: str, name: str) -> Path:
    return DATA_DIR / split / f"{split}_{name}.tsv"


def cache_path(name: str) -> Path:
    return CACHE_DIR / name
