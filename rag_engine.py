"""Core retrieval + RAG logic for the dry bulk shipping dashboard.

Ported from retrieving_article.ipynb's validated pipeline: hybrid BM25 +
dense retrieval with Reciprocal Rank Fusion and cross-encoder reranking for
article-level sources, plus chunk-level retrieval + a local Ollama LLM for
the grounded answer. All models/data load lazily and once (module-level
singletons), so the first request after startup is slow (loading models /
building indexes) and every request after that is fast.
"""
import random
import re
import string
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path

import faiss
import numpy as np
import ollama
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_system_prompt import RAG_REFUSAL_PHRASE, RAG_SYSTEM_PROMPT, RAG_USER_REMINDER

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_MODEL = "qwen3:4b-instruct"  # non-thinking variant; plain qwen3:4b thinks and is ~15x slower here

# Questions about "recent"/"lately"/"latest"/... are answered only from articles
# published in the latest calendar month that has data: today's month, or the
# month of the newest article if the scrape is behind. If that month has fewer
# than RECENT_MIN_ARTICLES articles or RECENT_MIN_MARKET_ARTICLES market reports,
# the window is widened back one month at a time and the page says so. A
# question that names a year other than the current one is not restricted.
RECENT_MIN_ARTICLES = 20
RECENT_MIN_MARKET_ARTICLES = 30
# An article counts as a market report if it mentions a Baltic index, FFAs or
# rate/earnings levels. Vessel class names alone don't count: sale and
# newbuilding news mentions them all the time.
MARKET_PATTERN = re.compile(
    r"\b(?:BDI|Baltic Dry|BCI|BPI|BSI|BHSI|C5TC|P[45]TC|S1[01]TC|HS7TC|FFAs?|forward freight|"
    r"(?:spot|freight|charter|timecharter|capesize|cape|panamax|kamsarmax|supramax|ultramax|handysize)"
    r" (?:rates?|earnings|market))\b|\$[\d,.]+k? ?(?:/|per |a )day",
    re.IGNORECASE,
)
# A recent question about the market itself searches only market reports, so
# company news that happens to say "dry bulk market" doesn't crowd them out.
MARKET_QUESTION_PATTERN = re.compile(r"\b(?:markets?|rates?|earnings|index|indices|freight)\b", re.IGNORECASE)
RECENCY_PATTERN = re.compile(
    r"\b(recent(ly)?|lately|latest|newest|current(ly)?|nowadays|these days|right now|"
    r"at the moment|at present|presently|this (week|month)|"
    r"(past|last) (few |couple of )?(days|weeks?|month))\b",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
OTHER_YEAR_PATTERN = re.compile(r"\b(last|previous|past) years?\b|\byears? ago\b", re.IGNORECASE)


def is_recency_query(query: str, today: pd.Timestamp | None = None) -> bool:
    if not RECENCY_PATTERN.search(query):
        return False
    year = (today or pd.Timestamp.today()).year
    asks_other_year = any(int(m.group()) != year for m in YEAR_PATTERN.finditer(query))
    return not (asks_other_year or OTHER_YEAR_PATTERN.search(query))


def strip_recency_words(query: str) -> str:
    """Drop "current", "lately", ... before searching; the date window already handles them."""
    stripped = re.sub(r"\s+", " ", RECENCY_PATTERN.sub(" ", query)).strip()
    return stripped if tokenize(stripped) else query

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "for", "to",
    "of", "in", "on", "at", "by", "with", "from", "as", "is", "are", "was",
    "were", "be", "been", "being", "it", "its", "this", "that", "these",
    "those", "i", "you", "he", "she", "we", "they", "them", "his", "her",
    "their", "our", "your", "not", "no", "so", "than", "too", "very", "can",
    "will", "would", "could", "should", "has", "have", "had", "do", "does",
    "did", "about", "into", "up", "down", "out", "over", "under", "again",
    "further", "all", "any", "both", "each", "few", "more", "most", "other",
    "how", "what", "whats", "which", "who", "why", "when", "where",
    "some", "such", "only", "own", "same",
}


# ---------------------------------------------------------------------------
# Data loading & cleaning (identical logic to the notebook, so the chunk
# order here matches the cached embeddings copied from that project).
# ---------------------------------------------------------------------------
def load_sources(data_dir: Path) -> pd.DataFrame:
    hellenic = pd.read_csv(data_dir / "Hellenic_df_full_with_articles.csv")
    splash = pd.read_csv(data_dir / "splash_df_full_with_articles.csv")

    breakwave = pd.read_csv(data_dir / "extracted_articles_new.csv")
    breakwave = breakwave.rename(columns={"Link": "link", "Title": "title", "Content": "text"})
    breakwave["source"] = "Breakwave"
    breakwave["author"] = None
    breakwave["date"] = breakwave["Date"]
    breakwave = breakwave[["source", "title", "date", "author", "link", "text"]]

    common_cols = ["source", "title", "date", "author", "link", "text"]
    return pd.concat(
        [hellenic[common_cols], splash[common_cols], breakwave[common_cols]],
        ignore_index=True,
    )


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    format_by_source = {
        "Hellenic": {"dayfirst": True},
        "Breakwave": {"dayfirst": False},
        "splash": {"dayfirst": False},
    }
    for source, kwargs in format_by_source.items():
        mask = df["source"] == source
        parsed.loc[mask] = pd.to_datetime(df.loc[mask, "date"], errors="coerce", **kwargs)
    df["parsed_date"] = parsed
    return df


def normalize_title(title: str) -> str:
    title = str(title).lower().strip()
    title = title.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", title)


def dedupe_articles(df: pd.DataFrame, title_sim_threshold: float = 0.92) -> pd.DataFrame:
    df = df.drop_duplicates(subset="link", keep="first")

    df["_norm_title"] = df["title"].map(normalize_title)
    df["_block_key"] = df["_norm_title"].str.split().str[:6].str.join(" ")

    keep_mask = np.ones(len(df), dtype=bool)
    df = df.reset_index(drop=True)

    for _, idxs in df.groupby("_block_key").groups.items():
        idxs = list(idxs)
        if len(idxs) < 2:
            continue
        for i in range(len(idxs)):
            if not keep_mask[idxs[i]]:
                continue
            for j in range(i + 1, len(idxs)):
                if not keep_mask[idxs[j]]:
                    continue
                ratio = SequenceMatcher(
                    None, df.loc[idxs[i], "_norm_title"], df.loc[idxs[j], "_norm_title"]
                ).ratio()
                if ratio >= title_sim_threshold:
                    keep_mask[idxs[j]] = False

    return df[keep_mask].drop(columns=["_norm_title", "_block_key"]).reset_index(drop=True)


def flag_short_text(df: pd.DataFrame, min_chars: int = 50) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["text"].fillna("")
    text_len = df["text"].str.len()
    return df[text_len >= min_chars].reset_index(drop=True)


def build_articles() -> pd.DataFrame:
    articles_raw = load_sources(DATA_DIR)
    articles_raw = parse_dates(articles_raw)
    articles_raw = dedupe_articles(articles_raw)
    articles = flag_short_text(articles_raw, min_chars=50)
    articles = articles.rename(columns={"parsed_date": "date_parsed"})
    articles["article_id"] = articles.index
    articles["search_text"] = articles["title"].fillna("") + ". " + articles["text"].fillna("")
    articles["is_market"] = articles["search_text"].str.contains(MARKET_PATTERN)
    return articles


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start:start + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunks(df: pd.DataFrame, text_field: str) -> pd.DataFrame:
    records = []
    for article_id, text in zip(df["article_id"], df[text_field]):
        for chunk in chunk_text(text):
            records.append({"article_id": article_id, "chunk_text": chunk})
    chunks_df = pd.DataFrame(records)
    chunks_df["chunk_id"] = chunks_df.index
    return chunks_df


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


def build_bm25_index(chunks_df: pd.DataFrame) -> BM25Okapi:
    tokenized = [tokenize(t) for t in chunks_df["chunk_text"]]
    return BM25Okapi(tokenized)


def bm25_search_chunks(bm25: BM25Okapi, chunks_df: pd.DataFrame, query: str) -> np.ndarray:
    return bm25.get_scores(tokenize(query))


def bm25_search_chunks_topk(bm25: BM25Okapi, chunks_df: pd.DataFrame, query: str, top_k: int = 100,
                            allowed_mask: np.ndarray | None = None):
    scores = bm25_search_chunks(bm25, chunks_df, query)
    chunk_ids = chunks_df["chunk_id"].values
    order = np.argsort(-scores)
    if allowed_mask is not None:
        order = order[allowed_mask[order]]
    order = order[:top_k]
    return [(int(chunk_ids[i]), float(scores[i])) for i in order]


def bm25_search_articles_topk(bm25: BM25Okapi, chunks_df: pd.DataFrame, query: str, top_k: int = 200,
                              allowed_mask: np.ndarray | None = None):
    scores = bm25_search_chunks(bm25, chunks_df, query)
    tmp = pd.DataFrame({"article_id": chunks_df["article_id"], "score": scores})
    if allowed_mask is not None:
        tmp = tmp[allowed_mask]
    article_scores = tmp.groupby("article_id")["score"].max()
    top = article_scores.sort_values(ascending=False).head(top_k)
    return list(top.items())


# ---------------------------------------------------------------------------
# Dense retrieval (bge-small-en-v1.5 + FAISS)
# ---------------------------------------------------------------------------
_embed_model = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed_texts(texts: list[str], is_query: bool = False, batch_size: int = 64) -> np.ndarray:
    model = get_embed_model()
    if is_query:
        texts = [BGE_QUERY_PREFIX + t for t in texts]
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=False,
        normalize_embeddings=True, convert_to_numpy=True,
    )
    return embeddings.astype("float32")


def build_or_load_dense_index(chunks_df: pd.DataFrame, cache_prefix: str, force_rebuild: bool = False):
    emb_path = CACHE_DIR / f"{cache_prefix}_embeddings.npy"
    index_path = CACHE_DIR / f"{cache_prefix}_faiss.index"

    if not force_rebuild and emb_path.exists() and index_path.exists():
        index = faiss.read_index(str(index_path))
        if index.ntotal == len(chunks_df):
            return index
        print(
            f"WARNING: cached index has {index.ntotal} vectors but current "
            f"data has {len(chunks_df)} chunks -- rebuilding."
        )

    embeddings = embed_texts(chunks_df["chunk_text"].tolist(), is_query=False)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    np.save(emb_path, embeddings)
    faiss.write_index(index, str(index_path))
    return index


def _faiss_search(index: faiss.Index, query_vec: np.ndarray, k: int, allowed_positions: np.ndarray | None):
    """Search the flat index, optionally restricted to the given row positions.
    Returns (scores, positions) with FAISS's -1 padding dropped."""
    if allowed_positions is None:
        scores, positions = index.search(query_vec, min(k, index.ntotal))
    else:
        selector = faiss.IDSelectorBatch(allowed_positions.astype("int64"))
        params = faiss.SearchParameters(sel=selector)
        scores, positions = index.search(query_vec, min(k, len(allowed_positions)), params=params)
    keep = positions[0] >= 0
    return scores[0][keep], positions[0][keep]


def dense_search_chunks_topk(index: faiss.Index, chunks_df: pd.DataFrame, query: str, top_k: int = 100,
                             allowed_positions: np.ndarray | None = None):
    query_vec = embed_texts([query], is_query=True)
    scores, positions = _faiss_search(index, query_vec, top_k, allowed_positions)
    chunk_ids = chunks_df.iloc[positions]["chunk_id"].values
    return [(int(cid), float(s)) for cid, s in zip(chunk_ids, scores)]


def dense_search_articles_topk(index: faiss.Index, chunks_df: pd.DataFrame, query: str, top_k: int = 200, search_k_chunks: int = 2000,
                               allowed_positions: np.ndarray | None = None):
    query_vec = embed_texts([query], is_query=True)
    scores, positions = _faiss_search(index, query_vec, search_k_chunks, allowed_positions)
    tmp = pd.DataFrame({
        "article_id": chunks_df.iloc[positions]["article_id"].values,
        "score": scores,
    })
    article_scores = tmp.groupby("article_id")["score"].max()
    top = article_scores.sort_values(ascending=False).head(top_k)
    return list(top.items())


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
def rrf_fuse(ranked_lists: list[list[tuple[int, float]]], k: int = 60, top_k: int = 100):
    rrf_scores: dict[int, float] = {}
    for ranked_list in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked_list, start=1):
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + 1.0 / (k + rank)
    fused = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)
    return fused[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder reranking
# ---------------------------------------------------------------------------
_rerank_model = None


def get_rerank_model() -> CrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = CrossEncoder(RERANK_MODEL_NAME)
    return _rerank_model


def rerank_article_snippet(row: pd.Series, n_words: int = 300) -> str:
    title = str(row["title"])
    body_words = str(row["text"]).split()[:n_words]
    return title + ". " + " ".join(body_words)


def rerank_articles(query: str, candidate_ids: list[int], df: pd.DataFrame) -> list[tuple[int, float]]:
    model = get_rerank_model()
    cand_df = df.set_index("article_id").loc[candidate_ids]
    pairs = [(query, rerank_article_snippet(row)) for _, row in cand_df.iterrows()]
    scores = model.predict(pairs, show_progress_bar=False)
    return sorted(zip(candidate_ids, scores), key=lambda kv: kv[1], reverse=True)


def rerank_chunks(query: str, chunk_ids: list[int], chunks_df: pd.DataFrame) -> list[tuple[int, float]]:
    model = get_rerank_model()
    cand = chunks_df.set_index("chunk_id").loc[chunk_ids]
    pairs = [(query, text) for text in cand["chunk_text"]]
    scores = model.predict(pairs, show_progress_bar=False)
    return sorted(zip(chunk_ids, scores), key=lambda kv: kv[1], reverse=True)


# ---------------------------------------------------------------------------
# Ollama generation
# ---------------------------------------------------------------------------
def ensure_ollama_model(model_name: str) -> None:
    local_models = {m.model for m in ollama.list().models}
    if model_name in local_models or f"{model_name}:latest" in local_models:
        return
    print(f"Pulling Ollama model '{model_name}' (one-time download)...")
    for _ in ollama.pull(model_name, stream=True):
        pass
    print(f"Done pulling '{model_name}'.")


def build_context_block(top_chunks: list[dict]) -> str:
    blocks = []
    for i, c in enumerate(top_chunks, start=1):
        date_str = c["date"].date() if pd.notna(c["date"]) else "unknown date"
        header = f"[{i}] {c['title']} ({c['source']}, {date_str})"
        blocks.append(f"{header}\n{c['chunk_text']}")
    return "\n\n".join(blocks)


def generate_answer(question: str, context_block: str, model: str = OLLAMA_MODEL,
                    recent: dict | None = None) -> tuple[str, float]:
    note = ""
    if recent is not None:
        note = (f"\n\nNote: this question is about recent developments. 'Recent' means {recent['label']}, "
                f"and all passages above were published in that period. Talk only about that period.")
        if recent["stale"]:
            note += (f" There are no newer articles: the latest is from {recent['latest'].strftime('%d %B %Y')}, "
                     f"so describe the market as of then and say so.")
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_block}\n\n{RAG_USER_REMINDER}{note}\n\nQuestion: {question}"},
    ]
    t0 = time.perf_counter()
    response = ollama.chat(model=model, messages=messages, options={"num_ctx": 8192, "num_predict": 400})  # hard cap on answer length
    latency_ms = (time.perf_counter() - t0) * 1000
    return response["message"]["content"], latency_ms


def extract_citations(answer_text: str) -> list[int]:
    return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer_text)})


# ---------------------------------------------------------------------------
# Engine: lazily built once, shared across requests.
# ---------------------------------------------------------------------------
class Engine:
    def __init__(self):
        self.articles: pd.DataFrame | None = None
        self.chunks: pd.DataFrame | None = None
        self.bm25: BM25Okapi | None = None
        self.faiss_index: faiss.Index | None = None
        self.chunk_dates: np.ndarray | None = None  # publication date of each chunk's article
        self.chunk_is_market: np.ndarray | None = None  # whether each chunk's article is a market report
        self.ready = False
        self.status = "not started"
        self._lock = threading.Lock()

    def build(self):
        with self._lock:
            if self.ready:
                return
            self.status = "loading & cleaning articles"
            self.articles = build_articles()

            self.status = "chunking articles"
            self.chunks = build_chunks(self.articles, "search_text")

            self.status = "building BM25 index"
            self.bm25 = build_bm25_index(self.chunks)

            self.status = "loading dense embeddings (reuses cached embeddings if chunk count matches)"
            self.faiss_index = build_or_load_dense_index(self.chunks, cache_prefix="prod")

            by_article = self.articles.set_index("article_id")
            self.chunk_dates = by_article["date_parsed"].reindex(self.chunks["article_id"]).to_numpy()
            self.chunk_is_market = by_article["is_market"].reindex(self.chunks["article_id"]).to_numpy()

            self.status = "warming up cross-encoder"
            get_rerank_model()

            self.status = "checking Ollama model"
            ensure_ollama_model(OLLAMA_MODEL)

            self.ready = True
            self.status = (f"ready ({len(self.articles)} articles, {len(self.chunks)} chunks; "
                           f"'recent' = {self.recent_window()['label']})")

    def build_async(self):
        threading.Thread(target=self.build, daemon=True).start()

    def recent_window(self, today: pd.Timestamp | None = None) -> dict:
        """Latest month with data, widened back a month at a time until it has enough
        articles and enough market reports."""
        today = (today or pd.Timestamp.today()).normalize()
        dates = self.articles["date_parsed"]
        is_market = self.articles["is_market"]
        latest = dates.max().normalize()
        month_start = min(today, latest).replace(day=1)
        end = month_start + pd.offsets.MonthBegin(1)   # exclusive

        def counts(start):
            in_window = (dates >= start) & (dates < end)
            return int(in_window.sum()), int((in_window & is_market).sum())

        start = month_start
        n_articles, n_market = counts(start)
        while ((n_articles < RECENT_MIN_ARTICLES or n_market < RECENT_MIN_MARKET_ARTICLES)
               and start > dates.min()):
            start -= pd.offsets.MonthBegin(1)
            n_articles, n_market = counts(start)

        last_month = month_start.strftime("%B %Y")
        label = last_month if start == month_start else f"{start.strftime('%B %Y')} to {last_month}"
        mask = (self.chunk_dates >= start) & (self.chunk_dates < end)
        return {
            "start": start, "end": end, "label": label,
            "requested": today.strftime("%B %Y"),
            "stale": latest < today.replace(day=1),   # no articles from this month yet
            "latest": latest,
            "fallback": start != month_start,
            "n_articles": n_articles,
            "n_market": n_market,
            "mask": mask,                       # bool per chunk row (BM25)
            "positions": np.flatnonzero(mask),  # FAISS row positions
        }

    # -- article-level hybrid search (for the "sources" list) --------------
    def search_articles(self, query: str, top_n: int = 8, bm25_k=200, dense_k=200, fused_k=50,
                        recent: dict | None = None) -> list[dict]:
        r = recent or {"mask": None, "positions": None}
        bm25_hits = bm25_search_articles_topk(self.bm25, self.chunks, query, top_k=bm25_k, allowed_mask=r["mask"])
        dense_hits = dense_search_articles_topk(self.faiss_index, self.chunks, query, top_k=dense_k,
                                                allowed_positions=r["positions"])
        fused = rrf_fuse([bm25_hits, dense_hits], k=60, top_k=fused_k)
        candidate_ids = [aid for aid, _ in fused]
        reranked = rerank_articles(query, candidate_ids, self.articles)[:top_n]

        results = []
        for aid, score in reranked:
            row = self.articles.set_index("article_id").loc[aid]
            results.append({
                "article_id": int(aid),
                "score": float(score),
                "title": row["title"],
                "date": row["date_parsed"],
                "source": row["source"],
                "link": row["link"],
            })
        # The most relevant articles are picked above; show them newest first.
        results.sort(key=lambda s: s["date"] if pd.notna(s["date"]) else pd.Timestamp.min, reverse=True)
        return results

    # -- chunk-level retrieval (for the LLM's grounded context) ------------
    def retrieve_top_chunks(self, query: str, final_k=5, bm25_k=100, dense_k=100, fused_k=30,
                            recent: dict | None = None) -> list[dict]:
        r = recent or {"mask": None, "positions": None}
        bm25_hits = bm25_search_chunks_topk(self.bm25, self.chunks, query, top_k=bm25_k, allowed_mask=r["mask"])
        dense_hits = dense_search_chunks_topk(self.faiss_index, self.chunks, query, top_k=dense_k,
                                              allowed_positions=r["positions"])
        fused = rrf_fuse([bm25_hits, dense_hits], k=60, top_k=fused_k)
        chunk_ids = [cid for cid, _ in fused]
        reranked = rerank_chunks(query, chunk_ids, self.chunks)[:final_k]

        results = []
        for cid, score in reranked:
            chunk_row = self.chunks.set_index("chunk_id").loc[cid]
            article = self.articles.set_index("article_id").loc[chunk_row["article_id"]]
            results.append({
                "chunk_id": int(cid),
                "score": float(score),
                "article_id": int(chunk_row["article_id"]),
                "chunk_text": chunk_row["chunk_text"],
                "title": article["title"],
                "date": article["date_parsed"],
                "source": article["source"],
                "link": article["link"],
            })
        return results

    def answer(self, question: str, n_sources: int = 8) -> dict:
        recent = self.recent_window() if is_recency_query(question) else None
        search_query = strip_recency_words(question) if recent else question
        if recent and MARKET_QUESTION_PATTERN.search(question):
            mask = recent["mask"] & self.chunk_is_market
            recent = {**recent, "mask": mask, "positions": np.flatnonzero(mask), "market_only": True}

        t0 = time.perf_counter()
        top_chunks = self.retrieve_top_chunks(search_query, final_k=5, recent=recent)
        sources = self.search_articles(search_query, top_n=n_sources, recent=recent)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        context_block = build_context_block(top_chunks)
        answer_text, generation_ms = generate_answer(question, context_block, recent=recent)
        cited = extract_citations(answer_text)

        return {
            "question": question,
            "recent": {k: v for k, v in recent.items() if k not in ("mask", "positions")} if recent else None,
            "answer": answer_text,
            "cited_chunk_numbers": cited,
            "context_chunks": top_chunks,
            "sources": sources,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
        }


engine = Engine()
