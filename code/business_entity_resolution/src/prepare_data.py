"""Stage 0: convert raw TSVs to Parquet (faster reloads, exact string preservation)."""
import time
import polars as pl
from config import raw_path, cache_path

SOURCES = ["source1", "source2", "source3"]


def read_tsv(path) -> pl.DataFrame:
    return pl.read_csv(
        path, separator="\t", quote_char=None, has_header=True,
        schema_overrides={c: pl.Utf8 for c in ["entity_id", "business_name",
                                                "business_address", "country"]},
    )


def main():
    for split in ["train", "test"]:
        for src in SOURCES:
            out = cache_path(f"{split}_{src}.parquet")
            if out.exists():
                print(f"skip {out.name} (exists)")
                continue
            t = time.time()
            p = raw_path(split, src)
            print(f"reading {p} ...", flush=True)
            df = read_tsv(p).with_columns(
                pl.col("business_name").fill_null(""),
                pl.col("business_address").fill_null(""),
                pl.col("country").fill_null(""),
            )
            df.write_parquet(out)
            print(f"{split}_{src}: {df.height:,} rows ({time.time()-t:.1f}s)", flush=True)
    # ground truth (train only)
    gt_out = cache_path("train_ground_truth.parquet")
    if not gt_out.exists():
        gt = pl.read_csv(
            raw_path("train", "ground_truth"), separator="\t", quote_char=None,
            schema_overrides={"source1_entity_id": pl.Utf8, "matched_entity_ids": pl.Utf8},
        ).with_columns(pl.col("matched_entity_ids").fill_null(""))
        gt.write_parquet(gt_out)
        print(f"ground truth: {gt.height:,} rows")
    else:
        print("skip train_ground_truth.parquet (exists)")


if __name__ == "__main__":
    main()
