# Dry Bulk Shipping RAG Dashboard

Ask a question about the dry bulk shipping market in plain English and get a short answer grounded in news articles, with numbered citations and links to the source articles.

The app is a small Flask web page on top of a retrieval-augmented generation (RAG) pipeline:

- **Hybrid search** over ~49,000 shipping news articles (BM25 keyword search + dense embeddings with FAISS), merged with Reciprocal Rank Fusion and re-ranked by a cross-encoder.
- **A local LLM** (`qwen3:4b-instruct` through [Ollama](https://ollama.com)) writes the answer using only the retrieved passages, and cites every claim.
- **Date-aware answers**: questions about "recent", "current" or "latest" developments only search the most recent period that has data, and the page says which period that was.
- **Everything runs locally**: no API keys and no data leaves your machine.

Example questions:

- *How is the current dry bulk market?*
- *Any recent capesize newbuilding orders?*
- *What drove Panamax rates in March 2025?*
- *Has the Baltic Dry Index been falling recently?*

---

## Contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Data](#data)
- [Running the app](#running-the-app)
- [Configuration](#configuration)
- [The system prompt](#the-system-prompt)
- [How "recent" questions are handled](#how-recent-questions-are-handled)
- [FFA vs spot notebook](#ffa-vs-spot-notebook)
- [Performance](#performance)
- [Limitations](#limitations)

---

## How it works

```
                    ┌──────────────────────────── startup (once) ────────────────────────────┐
 data/*.csv  ──►  load + parse dates  ──►  dedupe  ──►  drop short texts  ──►  chunk (200 words, 50 overlap)
                                                                                     │
                                                         ┌───────────────────────────┴───────────────┐
                                                         ▼                                           ▼
                                                  BM25 index (rank_bm25)             bge-small-en-v1.5 embeddings
                                                                                     → FAISS index (cached in cache/)
                    └────────────────────────────────────────────────────────────────────────────────┘

 question ──► recency check ──► BM25 top-k ─┐
                                            ├─► Reciprocal Rank Fusion ─► cross-encoder rerank ─┬─► top 5 chunks ─► Ollama LLM ─► answer + [n] citations
                               dense top-k ─┘                                                   └─► top N articles ─► "Sources" list
```

### 1. Data loading and cleaning (`rag_engine.py`)

- Three sources are loaded and put into one table with the columns `source, title, date, author, link, text`:
  Hellenic Shipping News, Splash247 and Breakwave Advisors.
- Dates are parsed per source (Hellenic uses day-first dates, the others month-first).
- Duplicates are removed: first by exact link, then by near-identical titles (titles sharing their first six words are compared with `difflib.SequenceMatcher`, threshold 0.92).
- Articles with less than 50 characters of text are dropped.
- Each article is flagged as a **market report** if it mentions a Baltic index (BDI, BCI, C5TC, ...), FFAs, rate or earnings levels, or a `$/day` figure. This flag is used for recent market questions (see below).

### 2. Chunking

Each article (`title + ". " + text`) is split into overlapping chunks of 200 words with a 50-word overlap. With the current data this gives ~92,000 chunks.

### 3. Indexing

| Index | Model / library | Notes |
|---|---|---|
| Keyword | `rank_bm25.BM25Okapi` | lower-cased alphanumeric tokens, stopwords removed |
| Dense | `BAAI/bge-small-en-v1.5` + `faiss.IndexFlatIP` | normalised embeddings (cosine similarity); queries get the BGE query prefix |

The FAISS index and embeddings are saved to `cache/prod_faiss.index` and `cache/prod_embeddings.npy`. On the next start they are reused if the number of chunks hasn't changed, and rebuilt automatically if it has.

### 4. Retrieval

Two retrieval passes run for every question:

| Pass | Purpose | BM25 top-k | Dense top-k | After fusion | Final |
|---|---|---|---|---|---|
| Chunk-level | context passages given to the LLM | 100 | 100 | 30 | 5 chunks |
| Article-level | "Sources" list on the page | 200 | 200 | 50 | 5–10 articles (user choice), newest first |

Both passes merge the BM25 and dense rankings with **Reciprocal Rank Fusion** (k = 60) and then re-rank the candidates with the cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2`. Article-level scores are the best score of any chunk in the article.

### 5. Answer generation

The top 5 chunks are numbered `[1]`–`[5]`, each with a header giving its title, source and publication date. They are sent to the local Ollama model with:

- the **system prompt** from `rag_system_prompt.py` (rules + dry bulk terminology reference), and
- a short **reminder** after the passages repeating the key rules (stay under 150 words, cite every claim, answer partially or refuse when the passages don't cover the question).

If nothing relevant is retrieved, the model answers with a fixed sentence:
*"The articles don't contain enough information to answer this question."*

The citation numbers in the answer are extracted so the page can link each `[n]` to its passage.

### 6. Web app (`app.py`)

- The data load and index build start in a background thread when the app starts, so the page opens right away and shows a "Warming up" status (polled from `/status`) until the engine is ready.
- The page shows the answer, the passages it used, the source articles with links, and retrieval/generation timings.

---

## Project structure

```
dry_bulk_rag_dashboard/
├── app.py                 # Flask app: routes, background engine start-up, /status endpoint
├── rag_engine.py          # data cleaning, chunking, BM25 + FAISS retrieval, RRF, reranking, Ollama call
├── rag_system_prompt.py   # system prompt, refusal sentence and the reminder added after the passages
├── requirements.txt
├── templates/
│   └── index.html         # the dashboard page (Jinja2)
├── static/
│   └── style.css
├── data/                  # input CSVs (not in the repo, see "Data")
├── cache/                 # FAISS index + embeddings, built automatically (not in the repo)
├── prompt_info/           # reference glossaries the terminology section of the prompt was written from (not in the repo)
├── FFA_spot.ipynb         # side analysis: FFA forward prices vs spot timecharter averages
└── FFA_with_Spot.csv      # output of that notebook (not in the repo)
```

---

## Requirements

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed and running
- About **6 GB of free disk space** (Python packages, embedding/reranking models, the Ollama model, the FAISS cache)
- **8 GB+ RAM**; 16 GB or more is more comfortable, since the dense index and all articles are held in memory
- A GPU is optional. With a 4 GB GPU (tested on a GTX 1650), answers take about 20 seconds; CPU-only is slower.

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/dry_bulk_rag_dashboard.git
cd dry_bulk_rag_dashboard

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 3. Install the Python dependencies
pip install -r requirements.txt

# 4. Download the LLM (optional: the app pulls it automatically on first start)
ollama pull qwen3:4b-instruct
```

The embedding model (`BAAI/bge-small-en-v1.5`) and the reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) are downloaded from Hugging Face automatically the first time the app runs.

Then put your article CSVs in `data/` (next section).

---

## Data

The article data is **not included** in this repository. It is scraped full-text news, which is copyrighted by the publishers. To run the app, put three CSV files in `data/`:

| File | Source | Required columns | Date format |
|---|---|---|---|
| `Hellenic_df_full_with_articles.csv` | Hellenic Shipping News | `source`, `title`, `date`, `author`, `link`, `text` | day-first (e.g. `09/09/2026`) |
| `splash_df_full_with_articles.csv` | Splash247 | `source`, `title`, `date`, `author`, `link`, `text` | month-first / ISO |
| `extracted_articles_new.csv` | Breakwave Advisors | `Title`, `Date`, `Link`, `Content` | month-first / ISO |

Notes:

- The `source` column must contain `Hellenic` and `splash` respectively (the date parser chooses the date format by source name). Breakwave's source name is added by the code.
- `author` may be empty.
- To use different sources, edit `load_sources()` and `parse_dates()` in `rag_engine.py`.

When the data changes, the FAISS cache is rebuilt automatically on the next start, because the chunk count no longer matches. You can also delete the `cache/` folder to force a rebuild.

---

## Running the app

1. Make sure Ollama is running (open the Ollama app, or run `ollama serve`).
2. Start the dashboard:

   ```bash
   python app.py
   ```

3. Open **http://127.0.0.1:5000** in your browser.
4. Type a question, choose how many source articles to show (5–10), and submit.

The first start is slow: the app loads and cleans all articles, builds the BM25 index, builds or loads the FAISS index, and loads the models. Building the embeddings from scratch (no cache) for ~92,000 chunks can take a long time on CPU. Later starts reuse the cache. The page shows the current step while it warms up.

Stop the app with `Ctrl+C` in the terminal.

---

## Configuration

All settings are constants at the top of `rag_engine.py`:

| Setting | Default | What it does |
|---|---|---|
| `OLLAMA_MODEL` | `qwen3:4b-instruct` | Ollama model used for answers. Use a non-thinking/instruct model: the plain `qwen3:4b` "thinks" first and was ~15× slower in testing. |
| `EMBED_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Dense embedding model. Changing it requires deleting `cache/`. |
| `RERANK_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used for re-ranking. |
| `RECENT_MIN_ARTICLES` | `20` | Minimum number of articles in the "recent" window. |
| `RECENT_MIN_MARKET_ARTICLES` | `30` | Minimum number of market reports in the "recent" window. |

LLM call options (in `generate_answer`):

| Option | Value | Why |
|---|---|---|
| `num_ctx` | `8192` | Context window large enough for the system prompt (~1,000 tokens) + 5 passages + the answer. Without it, Ollama's default can silently cut off the start of the prompt. |
| `num_predict` | `400` | Hard cap on answer length, so a small model can't run on for minutes. |

---

## The system prompt

`rag_system_prompt.py` holds three things:

- **`RAG_SYSTEM_PROMPT`**: rules plus a compact dry bulk **terminology reference** (vessel classes, Baltic indices and route codes, FFAs, chartering terms, port operations, regions, market language). The terminology helps a small model read the passages correctly, e.g. that index points and `$/day` are different units, or that `C5` (a route) is not `C5TC` (the Capesize average). The model is told never to cite the terminology or use it as a source of facts.
- **`RAG_REFUSAL_PHRASE`**: the exact sentence used when the passages contain nothing relevant.
- **`RAG_USER_REMINDER`**: the key rules repeated after the passages, where a small model is most likely to follow them.

Main rules the model must follow:

- answer only from the passages, and cite every claim (`[1]`, `[2]`, ...);
- when passages disagree, prefer the newest, and interpret "this week", "Friday" and similar against the passage's own date;
- quote numbers exactly with unit and period, with no unsupported calculations;
- keep **spot** (today's index or route levels) and **forward** (FFA prices for future months/quarters) separate;
- answer the covered part of a question and say what is missing, and refuse only when nothing relevant is present.

The terminology section was written from the glossaries in `prompt_info/` (chartering and maritime glossaries; not included in the repo).

---

## How "recent" questions are handled

A question counts as a *recent* question if it contains words like *recent, lately, latest, current, nowadays, this week, past few weeks*, and doesn't name a year other than the current one.

For these questions:

1. **The time window starts from the newest article, not today's date.** It is the latest calendar month that has data: today's month, or the month of the newest article if the data is behind.
2. **The window is widened** back one month at a time until it has at least `RECENT_MIN_ARTICLES` articles **and** `RECENT_MIN_MARKET_ARTICLES` market reports.
3. **Words like "current" and "lately" are removed** from the search query, since the date window already handles them.
4. **If the question is about the market** (it mentions *market, rates, index, earnings, freight*), only market reports in the window are searched, so company news that happens to say "dry bulk market" doesn't crowd them out.
5. The page states which months were searched, how many articles and market reports they contained, and, if the data is behind, the date of the newest article. The LLM is told the same, so it describes the market "as of" that date.

---

## FFA vs spot notebook

`FFA_spot.ipynb` is a separate analysis, not used by the app. It combines Baltic Forward Assessment (BFA) data for Capesize (5TC), Panamax (4TC) and Supramax (10TC) contracts with the spot timecharter averages (C5TC, P4TC, S10TC). It writes a daily table to `FFA_with_Spot.csv`: spot vs current-month, +1 month and +2 month forward prices, from 2010 onward.

It expects `data/bfa_cape.csv`, `data/bfa_panamax.csv`, `data/bfa_supramax.csv` and `data/Freight data.csv`. These are licensed Baltic Exchange data and are not included.

---

## Performance

Measured on a laptop with 24 GB RAM and a GTX 1650 (4 GB VRAM):

| Step | Time |
|---|---|
| Start-up with an existing cache | about a minute (loading ~49,000 articles, building BM25, loading the models) |
| Retrieval (both passes) | a few seconds |
| Answer generation (`qwen3:4b-instruct`, ~100–150 words) | ~15–25 seconds |

`qwen3:4b-instruct` doesn't fully fit in 4 GB of VRAM with an 8k context (about 20% runs on the CPU). A smaller model such as `qwen3:1.7b` would be faster but follows the citation and refusal rules less reliably.

---

## Limitations

- **Answers are only as current as the data.** If the scrape is behind, "current" questions are answered as of the newest article, and the page says so.
- **A 4B model makes mistakes.** It can occasionally misread a passage, for example attributing a vessel count to the wrong ship type. Check the cited passages for anything important; they are shown below every answer.
- **Market report detection is a keyword heuristic** (`MARKET_PATTERN` in `rag_engine.py`), so it can include articles that only mention a rate in passing.
- **Only 5 passages** are given to the LLM, so broad questions ("summarise 2025") get a narrow answer.
- **English only**, and tuned for dry bulk shipping terminology.
- **Not financial advice.** The dashboard summarises news coverage; it doesn't forecast markets.

---

## Acknowledgements

- News sources: [Hellenic Shipping News](https://www.hellenicshippingnews.com), [Splash247](https://splash247.com), [Breakwave Advisors](https://www.breakwaveadvisors.com)
- Models: [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2), [Qwen3](https://ollama.com/library/qwen3) via Ollama
- Libraries: Flask, pandas, NumPy, FAISS, rank-bm25, sentence-transformers
