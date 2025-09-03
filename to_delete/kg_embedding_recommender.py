#!/usr/bin/env python3
"""
KG-Embedding Recommender — v2 (TransE-style)

Features:
- Score a single *new* movie in embedding space (uses entity + relation CSVs)
- Batch-score many new movies from a CSV
- Cold-start evaluation: hold out a fraction of movies, build user vector on the rest,
  compose cold-start embeddings for held-out movies from neighbors, and report metrics.
- Frequency/evidence weighting: boosts recurring actors/directors/genres/lang seen in multiple liked/rated films.

Assumptions:
- Triples TSV columns: head, relation, tail (tab-separated)
- Ratings in "schema:review" as "personalVote_4.0" etc.
- Likes in (movie, ex:liked, liked_Yes)
- Names via "schema:name" to map human names -> entity IDs
- Relations: schema:actor, schema:director, schema:genre, ex:originalLanguage

"""

import argparse
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd


def parse_personal_vote(s: str) -> float:
    m = re.match(r"personalVote_([0-9]+(?:\.[0-9]+)?)", str(s))
    return float(m.group(1)) if m else float("nan")


def is_yes(s: str) -> bool:
    return str(s).lower() in {"liked_yes", "yes", "true", "1"}


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / (n + 1e-12)


def ndcg_at_k(relevances: List[float], k: int) -> float:
    k = min(k, len(relevances))
    if k == 0:
        return 0.0
    gains = np.array(relevances[:k], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.array(sorted(relevances, reverse=True)[:k], dtype=float)
    idcg = float(np.sum(ideal * discounts))
    return dcg / (idcg + 1e-12)


def hit_rate_at_k(relevances: List[int], k: int) -> float:
    k = min(k, len(relevances))
    return float(np.mean(relevances[:k])) if k > 0 else 0.0


class EmbeddingRecommender:
    def __init__(
            self,
            triples_path: str,
            entity_csv: str,
            relation_csv: str,
            like_bonus: float = 0.15,
            weight_floor: float = 0.2,
            rel_actor: str = "schema:actor",
            rel_director: str = "schema:director",
            rel_genre: str = "schema:genre",
            rel_language: str = "ex:originalLanguage",
            include_language: bool = True,
    ):
        # Load triples
        self.triples = pd.read_csv(triples_path, sep="\t", header=None, names=["head", "relation", "tail"])

        # Types and names
        types = self.triples[self.triples["relation"] == "rdf:type"]
        self.id2type: Dict[str, str] = dict(zip(types["head"], types["tail"]))
        names = self.triples[self.triples["relation"] == "schema:name"]
        self.id2name: Dict[str, str] = dict(zip(names["head"], names["tail"]))
        self.name2ids: Dict[str, List[str]] = {}
        for _id, nm in self.id2name.items():
            self.name2ids.setdefault(nm, []).append(_id)

        # Entities
        self.movies = {e for e, t in self.id2type.items() if t == "schema:Movie"}

        # Ratings & likes
        reviews = self.triples[self.triples["relation"] == "schema:review"]
        self.movie2rating: Dict[str, float] = {}
        for _, r in reviews.iterrows():
            mv = r["head"]
            if mv in self.movies:
                self.movie2rating[mv] = parse_personal_vote(r["tail"])
        likes = self.triples[self.triples["relation"] == "ex:liked"]
        self.movie2liked: Dict[str, bool] = {r["head"]: is_yes(r["tail"]) for _, r in likes.iterrows() if r["head"] in self.movies}

        # Normalize ratings
        ratings = pd.Series(self.movie2rating, dtype=float)
        if len(ratings) > 0:
            self.rmin = float(np.nanmin(ratings.values))
            self.rmax = float(np.nanmax(ratings.values))
            self.denom = (self.rmax - self.rmin) if self.rmax > self.rmin else 1.0
        else:
            self.rmin, self.rmax, self.denom = 0.0, 1.0, 1.0

        # Relations of interest
        self.rel_actor = rel_actor
        self.rel_director = rel_director
        self.rel_genre = rel_genre
        self.rel_language = rel_language
        self.include_language = include_language

        # Load embeddings
        self.E = pd.read_csv(entity_csv, index_col=0)
        self.R = pd.read_csv(relation_csv, index_col=0)
        self.dim = self.E.shape[1]

        # Precollect neighbor maps
        def collect(rel_name: str) -> Dict[str, List[str]]:
            m = defaultdict(list)
            for _, row in self.triples[self.triples["relation"] == rel_name].iterrows():
                mv = row["head"]
                if mv in self.movies:
                    m[mv].append(row["tail"])
            return m

        self.movie2actors = collect(self.rel_actor)
        self.movie2directors = collect(self.rel_director)
        self.movie2genres = collect(self.rel_genre)
        self.movie2langs = collect(self.rel_language)

        # Evidence maps for recurrence
        self.actor_evidence = self._build_role_evidence(self.movie2actors)
        self.director_evidence = self._build_role_evidence(self.movie2directors)
        self.genre_evidence = self._build_role_evidence(self.movie2genres)
        self.lang_evidence = self._build_role_evidence(self.movie2langs)

        self._cache_user_vec = None

    def _build_role_evidence(self, movie2role: Dict[str, List[str]], weight_floor: float = 0.2) -> Dict[str, float]:
        evid = {}
        for mv, roles in movie2role.items():
            w = self._movie_weight(mv)
            if w >= weight_floor:
                for rid in roles:
                    evid[rid] = evid.get(rid, 0.0) + w
        return evid

    def get_entity_vec(self, ent_id: str) -> Optional[np.ndarray]:
        if ent_id in self.E.index:
            return self.E.loc[ent_id].values.astype(np.float32)
        return None

    def get_relation_vec(self, rel: str) -> Optional[np.ndarray]:
        if rel in self.R.index:
            return self.R.loc[rel].values.astype(np.float32)
        return None

    def _movie_weight(self, mv: str) -> float:
        r = self.movie2rating.get(mv, float("nan"))
        w = 0.0
        if not (pd.isna(r)):
            w = (r - self.rmin) / self.denom  # 0..1
        if self.movie2liked.get(mv, False):
            w += 0.15
        return max(0.0, float(w))

    def build_user_vector(self, use_movies: Optional[Iterable[str]] = None, weight_floor: float = 0.2) -> np.ndarray:
        acc = []
        for mv in (use_movies if use_movies is not None else self.movies):
            if mv in self.E.index:
                w = self._movie_weight(mv)
                if w >= weight_floor:
                    acc.append(w * self.get_entity_vec(mv))
        if not acc:
            raise RuntimeError("No movies passed the weight floor or missing embeddings.")
        vec = np.sum(acc, axis=0)
        vec = l2_normalize(vec)
        self._cache_user_vec = vec
        return vec

    def translate_neighbor_to_movie(self, neighbor_id: str, rel_name: str) -> Optional[np.ndarray]:
        rel = self.get_relation_vec(rel_name)
        ent = self.get_entity_vec(neighbor_id)
        if ent is None:
            return None
        if rel is None:
            return ent
        return ent - rel

    def compose_new_movie_embedding(
            self,
            actors: Iterable[str] = (),
            directors: Iterable[str] = (),
            genres: Iterable[str] = (),
            languages: Iterable[str] = (),
    ) -> Optional[np.ndarray]:
        parts = []
        for a in actors:
            v = self.translate_neighbor_to_movie(a, self.rel_actor)
            if v is not None:
                parts.append(v)
        for d in directors:
            v = self.translate_neighbor_to_movie(d, self.rel_director)
            if v is not None:
                parts.append(v)
        for g in genres:
            v = self.translate_neighbor_to_movie(g, self.rel_genre)
            if v is not None:
                parts.append(v)
        if self.include_language:
            for l in languages:
                v = self.translate_neighbor_to_movie(l, self.rel_language)
                if v is not None:
                    parts.append(v)
        if not parts:
            return None
        return l2_normalize(np.mean(parts, axis=0))

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def ids_from_names(self, names: Iterable[str]) -> List[str]:
        out = []
        for nm in names:
            out.extend(self.name2ids.get(nm, []))
        return out

    def _apply_freq_boost(self, base_score: float, actor_ids, director_ids, genre_ids, lang_ids,
                          beta_actor=0.6, beta_director=0.7, beta_genre=0.4, beta_lang=0.2) -> float:
        boost = 1.0
        if actor_ids:
            boost *= (1.0 + beta_actor * np.log1p(np.mean([self.actor_evidence.get(a, 0.0) for a in actor_ids])))
        if director_ids:
            boost *= (1.0 + beta_director * np.log1p(np.mean([self.director_evidence.get(d, 0.0) for d in director_ids])))
        if genre_ids:
            boost *= (1.0 + beta_genre * np.log1p(np.mean([self.genre_evidence.get(g, 0.0) for g in genre_ids])))
        if lang_ids:
            boost *= (1.0 + beta_lang * np.log1p(np.mean([self.lang_evidence.get(l, 0.0) for l in lang_ids])))
        return base_score * boost

    def score_new_movie(self, actor_ids, director_ids, genre_ids, lang_ids, freq_boost=False,
                        beta_actor=0.6, beta_director=0.7, beta_genre=0.4, beta_lang=0.2):
        user_vec = self.build_user_vector()
        new_movie_vec = self.compose_new_movie_embedding(actor_ids, director_ids, genre_ids, lang_ids)
        if new_movie_vec is None:
            return None, None, None
        base_score = self.cosine(user_vec, new_movie_vec)
        final_score = base_score
        if freq_boost:
            final_score = self._apply_freq_boost(base_score, actor_ids, director_ids, genre_ids, lang_ids,
                                                 beta_actor, beta_director, beta_genre, beta_lang)
        return final_score, base_score, (final_score / (base_score + 1e-12))

    def score_batch_csv(self, csv_path: str, out_path: str, freq_boost=False,
                        beta_actor=0.6, beta_director=0.7, beta_genre=0.4, beta_lang=0.2) -> None:
        df = pd.read_csv(csv_path)

        def split_list(x):
            if pd.isna(x):
                return []
            s = str(x)
            if ";" in s:
                parts = [p.strip() for p in s.split(";")]
            elif "," in s:
                parts = [p.strip() for p in s.split(",")]
            else:
                parts = [s.strip()]
            return [p for p in parts if p]

        user_vec = self.build_user_vector()

        rows = []
        for _, row in df.iterrows():
            act_ids = self.ids_from_names(split_list(row.get("actors", "")))
            dir_ids = self.ids_from_names(split_list(row.get("directors", "")))
            gen_ids = self.ids_from_names(split_list(row.get("genres", "")))
            lang_ids = self.ids_from_names(split_list(row.get("lang", "")))

            new_vec = self.compose_new_movie_embedding(act_ids, dir_ids, gen_ids, lang_ids)
            score = self.cosine(user_vec, new_vec) if new_vec is not None else float("nan")
            if freq_boost:
                score = self._apply_freq_boost(score, act_ids, dir_ids, gen_ids, lang_ids,
                                               beta_actor, beta_director, beta_genre, beta_lang)

            rows.append({
                "title": row.get("title", ""),
                "score": score
            })

        pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", required=True)
    ap.add_argument("--entities", required=True)
    ap.add_argument("--relations", required=True)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--batch", default="")
    ap.add_argument("--out", default="scored_new_movies.csv")
    ap.add_argument("--actors", nargs="*", default=[])
    ap.add_argument("--directors", nargs="*", default=[])
    ap.add_argument("--genres", nargs="*", default=[])
    ap.add_argument("--lang", nargs="*", default=[])
    ap.add_argument("--freq-boost", action="store_true")
    ap.add_argument("--beta-actor", type=float, default=0.6)
    ap.add_argument("--beta-director", type=float, default=0.7)
    ap.add_argument("--beta-genre", type=float, default=0.4)
    ap.add_argument("--beta-lang", type=float, default=0.2)

    args = ap.parse_args()

    rec = EmbeddingRecommender(args.triples, args.entities, args.relations)

    if args.score:
        act_ids = rec.ids_from_names(args.actors)
        dir_ids = rec.ids_from_names(args.directors)
        gen_ids = rec.ids_from_names(args.genres)
        lang_ids = rec.ids_from_names(args.lang)

        final, base, boost = rec.score_new_movie(
            act_ids, dir_ids, gen_ids, lang_ids,
            freq_boost=args.freq_boost,
            beta_actor=args.beta_actor, beta_director=args.beta_director,
            beta_genre=args.beta_genre, beta_lang=args.beta_lang
        )
        if final is None:
            print("Could not compose movie vector.")
        else:
            print(f"Final score={final:.4f} (base cosine={base:.4f}, boost x{boost:.3f})")

    if args.batch:
        rec.score_batch_csv(
            args.batch, args.out,
            freq_boost=args.freq_boost,
            beta_actor=args.beta_actor, beta_director=args.beta_director,
            beta_genre=args.beta_genre, beta_lang=args.beta_lang
        )
        print(f"Batch results written to {args.out}")


if __name__ == "__main__":
    main()
