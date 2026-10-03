"""Stage 3 driver: compute pair features -> cache/<split>_feats/*.parquet

Disk-space optimized: reads cand files filtered to top K_FEAT candidates per query
(subsets the already-generated top-15 cands to top-10 at read time -- no re-retrieval needed).
Uses pyarrow incremental writer to stream batches without holding full file in RAM.
"""
import sys
import time

import polars as pl
import pyarrow.parquet as pq

from config import cache_path
from features import compute_features, s1_extra_table

# Filter cands to top-K at feature time (cand files have up to 15, we use 10)
# This cuts rows/file by 33% and is the main disk-space lever without re-running retrieval
K_FEAT = 10

SUB    = 400_000   # queries per sub-batch
Q_COLS = ["name_core", "name_norm", "addr_norm", "addr_nums",
          "is_domain", "has_indic", "addr_missing"]

COMPRESSION = "zstd"


def load_norm_tables(split: str):
    s1 = pl.read_parquet(cache_path(f"{split}_source1_norm.parquet"),
                         columns=["country", "name_core", "name_norm",
                                  "addr_norm", "addr_nums"]).with_row_index("s1_idx")
    q  = pl.concat([
        pl.read_parquet(cache_path(f"{split}_source2_norm.parquet"),
                        columns=Q_COLS).with_columns(pl.lit(False).alias("is_s3")),
        pl.read_parquet(cache_path(f"{split}_source3_norm.parquet"),
                        columns=Q_COLS).with_columns(pl.lit(True).alias("is_s3")),
    ]).with_row_index("q_idx")
    return s1, q


def build(split: str):
    cand_dir = cache_path(f"{split}_cands")
    out_dir  = cache_path(f"{split}_feats")
    out_dir.mkdir(exist_ok=True)
    s1, q = load_norm_tables(split)
    extra = s1_extra_table(s1, cand_dir)
    files = sorted(cand_dir.glob("*.parquet"))
    t0 = time.time()
    for f in files:
        out = out_dir / f.name
        if out.exists():
            continue
        tmp = out.with_suffix(".tmp")
        if tmp.exists():
            tmp.unlink()

        # Filter to top-K_FEAT per query at read time (saves ~33% disk vs K=15)
        c = (pl.read_parquet(f)
               .filter(pl.col("rank") <= K_FEAT)
               .sort("q_idx", "rank"))

        qids  = c["q_idx"].unique(maintain_order=True)
        writer = None
        total_rows = 0
        for s in range(0, len(qids), SUB):
            sub  = c.filter(pl.col("q_idx").is_in(qids.slice(s, SUB)))
            part = compute_features(sub, q, s1, extra)
            arrow_batch = part.to_arrow()
            del part, sub
            if writer is None:
                writer = pq.ParquetWriter(
                    str(tmp), arrow_batch.schema,
                    compression=COMPRESSION,
                )
            writer.write_table(arrow_batch)
            total_rows += len(arrow_batch)
            del arrow_batch
        if writer is not None:
            writer.close()
        tmp.replace(out)
        import shutil
        free_gb = shutil.disk_usage("C:").free // 1_000_000_000
        print(f"[{split}] {f.name}: {total_rows:,} rows, free disk {free_gb}GB "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    for sp_ in sys.argv[1:] or ["train", "test"]:
        build(sp_)