"""
Candidate generation (blocking) — retrieval layer.

Each S2/S3 record is a query; Source-1 records for the same country are the index.
Two sparse TF-IDF views:
  * name view  : character 3-grams of name_core with spaces removed
                 (robust to typos, token re-ordering, transliterated Indic names)
  * address view: word tokens from addr_norm (house numbers, street names, localities)

Combined score: w * cos_name + (1-w) * cos_addr.

For speed the concatenated (L2-normalised) sparse vector is compressed with a
CountSketch (random feature-hash) to `dim` dimensions and a brute-force fp16
top-K is run on GPU; the K approximate neighbours are then exact re-scored on
the original sparse vectors and the best k are kept.
Runs on CPU when CUDA is unavailable (slower candidate generation).
"""
import numpy as np
import polars as pl
import scipy.sparse as sp
import torch
from sklearn.feature_extraction.text import TfidfVectorizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def name_text(df: pl.DataFrame) -> list:
    # spaces removed so "dark buildwell" == "darkbuildwell"; padded for short names
    return [f"<{k}>" for k in df["name_core"].str.replace_all(" ", "").to_list()]


def rowwise_cosine(A, B, ia, ib, chunk=400_000):
    """cos(A[ia[i]], B[ib[i]]) for aligned index arrays (rows are L2-normalised)."""
    out = np.empty(len(ia), dtype=np.float32)
    for s in range(0, len(ia), chunk):
        x = A[ia[s:s+chunk]]
        y = B[ib[s:s+chunk]]
        out[s:s+chunk] = np.asarray(x.multiply(y).sum(axis=1)).ravel()
    return out


def _sketch_matrix(n_features: int, dim: int, seed: int) -> sp.csr_matrix:
    rng  = np.random.default_rng(seed)
    cols = rng.integers(0, dim, n_features)
    signs= rng.choice(np.array([-1.0, 1.0], dtype=np.float32), n_features)
    return sp.csr_matrix(
        (signs, (np.arange(n_features), cols)),
        shape=(n_features, dim), dtype=np.float32,
    )


class CountryIndex:
    """TF-IDF + GPU CountSketch index of the S1 records of one country."""

    def __init__(self, s1: pl.DataFrame, w_name=0.5, dim=1024, seed=42):
        self.ids    = s1["entity_id"].to_numpy()
        self.w_name = w_name
        self.dim    = dim
        # character 3-gram name index
        self.name_vec = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 3), min_df=2, max_df=0.05,
            sublinear_tf=True, dtype=np.float32,
        )
        # word address index
        self.addr_vec = TfidfVectorizer(
            analyzer="word", token_pattern=r"\S+", min_df=1, max_df=0.10,
            sublinear_tf=True, dtype=np.float32,
        )
        self.Xn = self.name_vec.fit_transform(name_text(s1))
        self.Xa = self.addr_vec.fit_transform(s1["addr_norm"].to_list())
        self.Pn = _sketch_matrix(self.Xn.shape[1], dim, seed)
        self.Pa = _sketch_matrix(self.Xa.shape[1], dim, seed + 1)
        self.D  = self._to_gpu(self._sketch(self.Xn, self.Xa))

    def encode(self, q: pl.DataFrame):
        return (
            self.name_vec.transform(name_text(q)),
            self.addr_vec.transform(q["addr_norm"].to_list()),
        )

    def _sketch(self, Xn, Xa):
        a = np.float32(np.sqrt(self.w_name))
        b = np.float32(np.sqrt(1.0 - self.w_name))
        return ((Xn @ self.Pn) * a + (Xa @ self.Pa) * b).astype(np.float32)

    @staticmethod
    def _to_gpu(S, chunk=20_000):
        out = torch.empty((S.shape[0], S.shape[1]), dtype=torch.float16, device=DEVICE)
        for s in range(0, S.shape[0], chunk):
            out[s:s+chunk] = torch.from_numpy(
                S[s:s+chunk].toarray().astype(np.float16)
            ).to(DEVICE)
        return out

    def approx_topk(self, Qn, Qa, K=50, batch=None):
        n = Qn.shape[0]
        if batch is None:
            batch = int(max(64, min(1024, 4e8 / (2 * self.D.shape[0]))))
        out = np.empty((n, K), dtype=np.int32)
        for s in range(0, n, batch * 16):
            Qs = self._sketch(Qn[s:s+batch*16], Qa[s:s+batch*16])
            for t in range(0, Qs.shape[0], batch):
                q  = self._to_gpu(Qs[t:t+batch])
                scores = q @ self.D.T
                out[s+t:s+t+q.shape[0]] = scores.topk(K, dim=1).indices.cpu().numpy()
        return out

    def exact_scores(self, Qn, Qa, qi, di):
        cn = rowwise_cosine(Qn, self.Xn, qi, di)
        ca = rowwise_cosine(Qa, self.Xa, qi, di)
        return cn, ca

    def candidates(self, Qn, Qa, K=50, k=10) -> pl.DataFrame:
        """Approximate top-K on GPU, exact re-score, keep best k per query.
        Returns DataFrame(q_row, s1_row, cos_name, cos_addr, cos_comb, rank)."""
        nb = self.approx_topk(Qn, Qa, K)
        qi = np.repeat(np.arange(Qn.shape[0], dtype=np.int32), K)
        di = nb.ravel()
        cn, ca = self.exact_scores(Qn, Qa, qi, di)
        df = pl.DataFrame({"q_row": qi, "s1_row": di,
                           "cos_name": cn, "cos_addr": ca})
        df = df.with_columns(
            (self.w_name * pl.col("cos_name") +
             (1 - self.w_name) * pl.col("cos_addr")).alias("cos_comb")
        )
        df = df.with_columns(
            pl.col("cos_comb").rank("ordinal", descending=True)
              .over("q_row").cast(pl.Int16).alias("rank")
        )
        return df.filter(pl.col("rank") <= k)
