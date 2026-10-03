"""
End-to-end pipeline runner: raw TSVs -> outputs/matching_results.tsv + candidate_pairs.tsv

Run from the pipeline/ directory:
    cd pipeline
    python run_all.py

Each stage caches its result and is skipped/resumed when the cache exists.
Stages:
  0  prepare_data      TSV -> Parquet
  1  build_normalized  name + address normalization (multiprocess)
  1b indic_dictionary  learn Indic token dict from training GT
  1c apply_indic_dict  apply dict to S2/S3 norm tables
  2  candidates        GPU TF-IDF + CountSketch ANN candidate pairs
  3  build_features    57 pairwise features per candidate pair
  4  train             LightGBM training + F0.5 threshold sweep
  5  predict           score test, write TSV outputs
"""
import time

import apply_indic_dict
import build_features
import build_normalized
import candidates
import indic_dictionary
import predict
import prepare_data
import train
from config import cache_path


def main():
    t0 = time.time()

    # Stage 0: TSV -> Parquet
    if not cache_path("test_source3.parquet").exists():
        prepare_data.main()
    else:
        print("Stage 0: Parquet files already exist, skipping.")

    # Stage 1: normalize names + addresses
    build_normalized.main(("train", "test"))

    # Stage 1b/c: Indic token dictionary
    if not indic_dictionary.DICT_PATH.exists():
        indic_dictionary.learn()
    mapping = indic_dictionary.load()
    for split in ("train", "test"):
        apply_indic_dict.apply(split, mapping)

    # Stage 2: candidate generation
    for split in ("train", "test"):
        candidates.generate(split)

    # Stage 3: feature computation
    for split in ("train", "test"):
        build_features.build(split)

    # Stage 4: train LightGBM
    if not train.MODEL_PATH.exists():
        train.main()
    else:
        print("Stage 4: model already trained, skipping.")

    # Stage 5: test inference -> TSV outputs
    predict.main()

    elapsed = (time.time() - t0) / 60
    print(f"\nPipeline finished in {elapsed:.1f} min")
    print("Outputs:")
    from config import OUTPUT_DIR
    for f in sorted(OUTPUT_DIR.glob("*.tsv")):
        size = f.stat().st_size / 1e6
        print(f"  {f}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
