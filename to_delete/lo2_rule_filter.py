import pandas as pd
import re

def _norm_title(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.lower().strip()
    # remove punctuation-like characters and extra spaces
    t = re.sub(r"[\W_]+", " ", t).strip()
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    return t

def _coerce_int(x):
    try:
        return int(x)
    except Exception:
        return None

def build_seen_pairs(watched_df, title_col="Name", year_col="Year"):
    """Return a set of (norm_title, year) pairs from Letterboxd watched.csv."""
    tmp = watched_df.copy()
    tmp["__norm_title__"] = tmp[title_col].map(_norm_title)
    tmp["__year__"] = tmp[year_col].map(_coerce_int)
    return set(zip(tmp["__norm_title__"], tmp["__year__"]))

def filter_unseen_topk(candidates_df, watched_df, k=5,
                       cand_title_col="candidate_title", cand_year_col="year",
                       sort_by="final", ascending=False):
    """
    Apply the logical rule:
      recommend(u,m) :- candidateFor(m,u) AND NOT watched(u,m)
    and return the top-k unseen candidates, sorted by `sort_by`.
    """
    # Build seen lookup
    seen_pairs = build_seen_pairs(watched_df, title_col="Name", year_col="Year")

    # Normalize candidate titles/years
    df = candidates_df.copy()
    df["__norm_title__"] = df[cand_title_col].map(_norm_title)
    df["__year__"] = df[cand_year_col].map(_coerce_int)

    # Drop duplicate (title, year) candidates, keep best by score
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)
    df = df.drop_duplicates(subset=["__norm_title__","__year__"], keep="first")

    # Filter out seen
    df["__seen__"] = list(zip(df["__norm_title__"], df["__year__"]))
    df = df[~df["__seen__"].isin(seen_pairs)]

    # Keep top-k
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending)

    result = df.head(k).drop(columns=["__norm_title__","__year__","__seen__"])
    if len(result) < k:
        raise ValueError(f"Only {len(result)} unseen candidates available. "
                         f"Increase your upstream candidate pool (e.g., top-N) to ensure {k} results.")
    return result

def apply_unseen_rule_and_write(candidates_df, watched_df, out_path,
                                k=5, cand_title_col="candidate_title", cand_year_col="year",
                                sort_by="final", ascending=False):
    """Filter candidates with the LO2 rule and write exactly k rows to CSV at out_path."""
    result = filter_unseen_topk(candidates_df, watched_df, k=k,
                                cand_title_col=cand_title_col, cand_year_col=cand_year_col,
                                sort_by=sort_by, ascending=ascending)
    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False)
    return result
