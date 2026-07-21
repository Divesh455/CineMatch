import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =========================
# ENV
# =========================
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    # Don't crash import-time in production if you prefer; but for you better fail early:
    raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")


# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="Movie Recommender API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for local streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# PICKLE GLOBALS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
if not os.path.exists(MODELS_DIR):
    PARENT_DIR = os.path.dirname(BASE_DIR)
    MODELS_DIR = os.path.join(PARENT_DIR, "Similarity Model")

DF_PATH = os.path.join(MODELS_DIR, "movie_info.pkl")
INDICES_PATH = os.path.join(MODELS_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(MODELS_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(MODELS_DIR, "tfidf.pkl")

df: Optional[pd.DataFrame] = None
indices_obj: Any = None
tfidf_matrix: Any = None
tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =========================
# MODELS
# =========================
class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =========================
# UTILS
# =========================
def _norm_title(t: str) -> str:
    return str(t).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    return f"{TMDB_IMG_500}{path}"


async def tmdb_get(path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Safe TMDB GET:
    - Network errors -> 502
    - TMDB API errors -> 502 with detail
    """
    q = dict(params)
    q["api_key"] = TMDB_API_KEY

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(f"{TMDB_BASE}{path}", params=q)
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}",
        )

    if r.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"TMDB error {r.status_code}: {r.text}"
        )

    return r.json()


async def tmdb_cards_from_results(
    results: List[dict], limit: int = 20
) -> List[TMDBMovieCard]:
    out: List[TMDBMovieCard] = []
    for m in (results or [])[:limit]:
        out.append(
            TMDBMovieCard(
                tmdb_id=int(m["id"]),
                title=m.get("title") or m.get("name") or "",
                poster_url=make_img_url(m.get("poster_path")),
                release_date=m.get("release_date"),
                vote_average=m.get("vote_average"),
            )
        )
    return out


async def tmdb_movie_details(movie_id: int) -> TMDBMovieDetails:
    data = await tmdb_get(f"/movie/{movie_id}", {"language": "en-US"})
    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or [],
    )


async def tmdb_search_movies(query: str, page: int = 1) -> Dict[str, Any]:
    """
    Raw TMDB response for keyword search (MULTIPLE results).
    Streamlit will use this for suggestions and grid.
    """
    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page,
        },
    )


async def tmdb_search_first(query: str) -> Optional[dict]:
    data = await tmdb_search_movies(query=query, page=1)
    results = data.get("results", [])
    return results[0] if results else None


# =========================
# TF-IDF Helpers
# =========================
def build_title_to_idx_map(indices: Any) -> Dict[str, int]:
    """
    indices.pkl can be:
    - dict(title -> index)
    - pandas Series (index=title, value=index)
    We normalize into TITLE_TO_IDX.
    """
    title_to_idx: Dict[str, int] = {}

    if isinstance(indices, dict):
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx

    # pandas Series or similar mapping
    try:
        for k, v in indices.items():
            title_to_idx[_norm_title(k)] = int(v)
        return title_to_idx
    except Exception:
        # last resort: if it's a list-like etc.
        raise RuntimeError(
            "indices.pkl must be dict or pandas Series-like (with .items())"
        )


def get_local_idx_by_title(title: str) -> Optional[int]:
    global TITLE_TO_IDX
    if not TITLE_TO_IDX:
        return None
    key = _norm_title(title)
    if key in TITLE_TO_IDX:
        return int(TITLE_TO_IDX[key])
    return None


def tfidf_recommend_titles(
    query_title: str, top_n: int = 10
) -> List[Tuple[str, float]]:
    """
    Returns list of (title, score) from local df using cosine similarity on TF-IDF matrix.
    Safe against missing columns/rows or unpickling failures.
    """
    global df, tfidf_matrix
    if df is None or tfidf_matrix is None:
        return []

    idx = get_local_idx_by_title(query_title)
    if idx is None:
        return []

    try:
        qv = tfidf_matrix[idx]
        scores = (tfidf_matrix @ qv.T).toarray().ravel()
        order = np.argsort(-scores)

        out: List[Tuple[str, float]] = []
        for i in order:
            if int(i) == int(idx):
                continue
            try:
                title_i = str(df.iloc[int(i)]["title"])
            except Exception:
                continue
            out.append((title_i, float(scores[int(i)])))
            if len(out) >= top_n:
                break
        return out
    except Exception as e:
        print(f"TF-IDF recommendation error: {e}")
        return []


async def attach_tmdb_card_by_title(title: str) -> Optional[TMDBMovieCard]:
    """
    Uses TMDB search by title to fetch poster for a local title.
    If not found, returns None (never crashes the endpoint).
    """
    try:
        m = await tmdb_search_first(title)
        if not m:
            return None
        return TMDBMovieCard(
            tmdb_id=int(m["id"]),
            title=m.get("title") or title,
            poster_url=make_img_url(m.get("poster_path")),
            release_date=m.get("release_date"),
            vote_average=m.get("vote_average"),
        )
    except Exception:
        return None


# =========================
# STARTUP: LOAD PICKLES
# =========================
@app.on_event("startup")
def load_pickles():
    global df, indices_obj, tfidf_matrix, tfidf_obj, TITLE_TO_IDX

    try:
        if not os.path.exists(DF_PATH):
            print(f"Pickle file not found at {DF_PATH}")
            return

        with open(DF_PATH, "rb") as f:
            raw_df = pickle.load(f)
            if isinstance(raw_df, pd.DataFrame):
                df = raw_df
            elif hasattr(raw_df, "__self__") and isinstance(raw_df.__self__, pd.DataFrame):
                df = raw_df.__self__
            elif callable(raw_df):
                df = pd.DataFrame(raw_df())
            elif isinstance(raw_df, dict):
                df = pd.DataFrame(raw_df)
            else:
                df = raw_df

        if os.path.exists(INDICES_PATH):
            with open(INDICES_PATH, "rb") as f:
                indices_obj = pickle.load(f)
            TITLE_TO_IDX = build_title_to_idx_map(indices_obj)

        if os.path.exists(TFIDF_MATRIX_PATH):
            with open(TFIDF_MATRIX_PATH, "rb") as f:
                tfidf_matrix = pickle.load(f)

        if os.path.exists(TFIDF_PATH):
            with open(TFIDF_PATH, "rb") as f:
                tfidf_obj = pickle.load(f)

        print("Successfully loaded model pickles.")
    except Exception as e:
        print(f"Warning: Exception while loading pickles: {e}")


# =========================
# FRONTEND HTML
# =========================
INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>CineMatch - Movie Recommender</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090c15;
      --card-bg: rgba(19, 24, 38, 0.7);
      --card-hover: rgba(29, 36, 56, 0.95);
      --accent: #6366f1;
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
      --text: #f3f4f6;
      --text-muted: #9ca3af;
      --border: rgba(255, 255, 255, 0.08);
      --modal-bg: rgba(12, 16, 26, 0.96);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    body { background-color: var(--bg); color: var(--text); min-height: 100vh; overflow-x: hidden; }

    /* Glassmorphism Header */
    header {
      position: sticky; top: 0; z-index: 100;
      background: rgba(9, 12, 21, 0.85); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
      border-bottom: 1px solid var(--border); padding: 1.1rem 2.5rem;
      display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap;
    }
    .logo {
      font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px;
      background: var(--accent-gradient); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      display: flex; align-items: center; gap: 0.5rem; cursor: pointer; user-select: none;
    }

    .search-box { position: relative; flex: 1; max-width: 540px; }
    .search-input {
      width: 100%; padding: 0.8rem 1.4rem; border-radius: 9999px;
      background: rgba(255,255,255,0.04); border: 1px solid var(--border);
      color: var(--text); font-size: 0.95rem; outline: none; transition: all 0.25s ease;
    }
    .search-input:focus { border-color: var(--accent); box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.2); background: rgba(255,255,255,0.07); }

    .suggestions {
      position: absolute; top: 115%; left: 0; right: 0; background: #131826;
      border: 1px solid var(--border); border-radius: 16px; max-height: 380px; overflow-y: auto;
      box-shadow: 0 25px 50px -12px rgba(0,0,0,0.8); z-index: 200; display: none; padding: 0.4rem;
    }
    .suggestion-item {
      padding: 0.75rem 1rem; border-radius: 10px; cursor: pointer; transition: background 0.15s;
      display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .suggestion-item:last-child { border-bottom: none; }
    .suggestion-item:hover { background: rgba(99, 102, 241, 0.2); }
    .suggestion-title { font-size: 0.92rem; font-weight: 600; color: #fff; }
    .suggestion-meta { font-size: 0.82rem; color: var(--text-muted); }

    main { max-width: 1400px; margin: 0 auto; padding: 2rem 2.5rem; }

    /* Category Nav Pills */
    .nav-pills { display: flex; gap: 0.8rem; margin-bottom: 2rem; overflow-x: auto; padding-bottom: 0.5rem; scrollbar-width: none; }
    .nav-pills::-webkit-scrollbar { display: none; }
    .pill {
      padding: 0.7rem 1.4rem; border-radius: 9999px; background: rgba(255,255,255,0.04);
      border: 1px solid var(--border); color: var(--text-muted); cursor: pointer;
      font-weight: 600; font-size: 0.9rem; transition: all 0.2s; white-space: nowrap;
    }
    .pill:hover, .pill.active { background: var(--accent-gradient); color: #fff; border-color: transparent; box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35); }

    .section-title { font-size: 1.45rem; font-weight: 700; letter-spacing: -0.3px; margin-bottom: 1.5rem; }

    /* Movie Cards Grid */
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 1.6rem; }

    .card {
      background: var(--card-bg); border-radius: 18px; overflow: hidden;
      border: 1px solid var(--border); cursor: pointer; transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
      display: flex; flex-direction: column; position: relative;
    }
    .card:hover { transform: translateY(-8px) scale(1.02); background: var(--card-hover); border-color: rgba(99, 102, 241, 0.5); box-shadow: 0 20px 35px -10px rgba(0,0,0,0.8); }
    .card-img-wrap { width: 100%; aspect-ratio: 2/3; background: #131826; position: relative; overflow: hidden; }
    .card-img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.4s ease; }
    .card:hover .card-img { transform: scale(1.08); }
    .no-img { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 2rem; }

    .rating-badge {
      position: absolute; top: 10px; right: 10px; background: rgba(9, 12, 21, 0.85);
      backdrop-filter: blur(8px); padding: 4px 8px; border-radius: 8px; font-size: 0.78rem; font-weight: 700; color: #fbbf24; border: 1px solid rgba(255,255,255,0.12);
    }

    .card-info { padding: 1.1rem; display: flex; flex-direction: column; gap: 0.35rem; flex: 1; }
    .card-title { font-size: 0.95rem; font-weight: 700; line-height: 1.3; color: var(--text); display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .card-meta { font-size: 0.82rem; color: var(--text-muted); }

    /* Spinner */
    .spinner { border: 3px solid rgba(255,255,255,0.08); border-left-color: var(--accent); border-radius: 50%; width: 44px; height: 44px; animation: spin 0.9s linear infinite; margin: 4rem auto; grid-column: 1/-1; }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    /* Modal */
    .modal {
      position: fixed; inset: 0; z-index: 300; background: rgba(0,0,0,0.85); backdrop-filter: blur(16px);
      display: none; align-items: center; justify-content: center; padding: 2rem; overflow-y: auto;
    }
    .modal-content {
      background: var(--modal-bg); border: 1px solid var(--border); border-radius: 26px;
      max-width: 1050px; width: 100%; max-height: 90vh; overflow-y: auto; position: relative;
      box-shadow: 0 30px 60px -15px rgba(0,0,0,0.9); scrollbar-width: thin;
    }
    .modal-close {
      position: absolute; top: 20px; right: 24px; z-index: 10; background: rgba(255,255,255,0.1);
      border: 1px solid var(--border); color: #fff; width: 38px; height: 38px; border-radius: 50%; cursor: pointer;
      font-size: 1.1rem; transition: all 0.2s; display: flex; align-items: center; justify-content: center;
    }
    .modal-close:hover { background: rgba(239, 68, 68, 0.8); transform: rotate(90deg); }

    .backdrop-container { width: 100%; height: 280px; position: relative; background-size: cover; background-position: center; border-bottom: 1px solid var(--border); }
    .backdrop-overlay { position: absolute; inset: 0; background: linear-gradient(to bottom, transparent 20%, var(--modal-bg) 100%); }

    .details-header { display: flex; gap: 2rem; padding: 0 2.5rem 2rem 2.5rem; margin-top: -90px; position: relative; z-index: 2; flex-wrap: wrap; }
    .poster-large { width: 185px; border-radius: 16px; border: 2px solid var(--border); box-shadow: 0 15px 35px rgba(0,0,0,0.6); object-fit: cover; aspect-ratio: 2/3; background: #131826; flex-shrink: 0; }

    .details-body { flex: 1; min-width: 280px; display: flex; flex-direction: column; gap: 0.75rem; justify-content: flex-end; }
    .details-title { font-size: 2.2rem; font-weight: 800; color: #fff; line-height: 1.15; }
    .genres-list { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 4px; }
    .genre-tag { background: rgba(99, 102, 241, 0.18); border: 1px solid rgba(99, 102, 241, 0.4); color: #a5b4fc; padding: 3px 12px; border-radius: 9999px; font-size: 0.82rem; font-weight: 600; }
    .overview-text { color: var(--text-muted); line-height: 1.7; font-size: 0.98rem; margin-top: 0.5rem; }

    .recs-section { padding: 2rem 2.5rem 2.5rem 2.5rem; border-top: 1px solid var(--border); }

    @media (max-width: 768px) {
      header { padding: 1rem; }
      main { padding: 1.2rem; }
      .details-header { margin-top: -50px; padding: 0 1.5rem 1.5rem 1.5rem; }
      .poster-large { width: 140px; }
      .recs-section { padding: 1.5rem; }
    }
  </style>
</head>
<body>

  <header>
    <div class="logo" onclick="loadCategory('popular', document.querySelector('.pill'))">
      🎬 CineMatch
    </div>
    
    <div class="search-box">
      <input type="text" class="search-input" id="searchInput" placeholder="Search movies (e.g. Inception, Batman, Avatar)..." autocomplete="off">
      <div class="suggestions" id="suggestionsBox"></div>
    </div>
  </header>

  <main>
    <!-- Category Filter Pills -->
    <div class="nav-pills">
      <button class="pill active" onclick="loadCategory('popular', this)">🔥 Popular</button>
      <button class="pill" onclick="loadCategory('trending', this)">📈 Trending Today</button>
      <button class="pill" onclick="loadCategory('top_rated', this)">⭐ Top Rated</button>
      <button class="pill" onclick="loadCategory('upcoming', this)">🚀 Upcoming</button>
    </div>

    <div class="section-title" id="gridTitle">🔥 Popular Movies</div>
    <div class="grid" id="movieGrid"></div>
  </main>

  <!-- Movie Details & Recommendations Modal -->
  <div class="modal" id="movieModal">
    <div class="modal-content">
      <button class="modal-close" onclick="closeModal()">✕</button>

      <div class="backdrop-container" id="modalBackdrop">
        <div class="backdrop-overlay"></div>
      </div>

      <div class="details-header">
        <img class="poster-large" id="modalPoster" src="" alt="Poster">
        <div class="details-body">
          <h2 class="details-title" id="modalTitle">Movie Title</h2>
          <div class="card-meta" id="modalMeta">Release Date</div>
          <div class="genres-list" id="modalGenres"></div>
          <p class="overview-text" id="modalOverview">Overview text...</p>
        </div>
      </div>

      <div class="recs-section">
        <div class="section-title">🔎 Similar Movies (TF-IDF Similarity)</div>
        <div class="grid" id="tfidfGrid" style="margin-bottom: 2.5rem;"></div>

        <div class="section-title">🎭 More Like This (Genre Match)</div>
        <div class="grid" id="genreGrid"></div>
      </div>

    </div>
  </div>

  <script>
    let debounceTimer;

    document.addEventListener('DOMContentLoaded', () => {
      loadCategory('popular', document.querySelector('.pill.active'));

      const searchInput = document.getElementById('searchInput');
      const suggestionsBox = document.getElementById('suggestionsBox');

      searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        const val = e.target.value.trim();
        if (val.length < 2) {
          suggestionsBox.style.display = 'none';
          return;
        }
        debounceTimer = setTimeout(() => fetchSuggestions(val), 250);
      });

      document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !suggestionsBox.contains(e.target)) {
          suggestionsBox.style.display = 'none';
        }
      });
    });

    async function loadCategory(cat, btn) {
      if (btn) {
        document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
      }

      const grid = document.getElementById('movieGrid');
      const title = document.getElementById('gridTitle');
      title.innerText = btn ? btn.innerText + ' Movies' : 'Movies';
      grid.innerHTML = '<div class="spinner"></div>';

      try {
        const res = await fetch(`/home?category=${cat}&limit=24`);
        const data = await res.json();
        renderCards(grid, data);
      } catch (err) {
        grid.innerHTML = `<div style="color:#ef4444; grid-column:1/-1;">Failed to load movies: ${err}</div>`;
      }
    }

    async function fetchSuggestions(query) {
      try {
        const res = await fetch(`/tmdb/search?query=${encodeURIComponent(query)}`);
        const data = await res.json();
        const results = data.results || [];
        const box = document.getElementById('suggestionsBox');

        if (!results.length) {
          box.style.display = 'none';
          return;
        }

        box.innerHTML = results.slice(0, 8).map(m => `
          <div class="suggestion-item" onclick="openMovieDetails(${m.id}, '${escapeHtml(m.title)}')">
            <span class="suggestion-title">${escapeHtml(m.title)}</span>
            <span class="suggestion-meta">${(m.release_date || '').slice(0,4)}</span>
          </div>
        `).join('');
        box.style.display = 'block';
      } catch (e) { console.error(e); }
    }

    function renderCards(container, cards) {
      if (!cards || !cards.length) {
        container.innerHTML = '<div style="grid-column: 1/-1; text-align:center; color:var(--text-muted); padding: 3rem 0;">No movies found.</div>';
        return;
      }

      container.innerHTML = cards.map(c => `
        <div class="card" onclick="openMovieDetails(${c.tmdb_id}, '${escapeHtml(c.title)}')">
          <div class="card-img-wrap">
            ${c.poster_url ? `<img class="card-img" src="${c.poster_url}" alt="${escapeHtml(c.title)}" loading="lazy">` : `<div class="no-img">🎬</div>`}
            ${c.vote_average ? `<div class="rating-badge">⭐ ${c.vote_average.toFixed(1)}</div>` : ''}
          </div>
          <div class="card-info">
            <div class="card-title">${escapeHtml(c.title)}</div>
            <div class="card-meta">${(c.release_date || '').slice(0,4)}</div>
          </div>
        </div>
      `).join('');
    }

    async function openMovieDetails(tmdbId, title) {
      document.getElementById('suggestionsBox').style.display = 'none';
      const modal = document.getElementById('movieModal');
      modal.style.display = 'flex';

      document.getElementById('modalTitle').innerText = title || 'Loading...';
      document.getElementById('modalMeta').innerText = '';
      document.getElementById('modalGenres').innerHTML = '';
      document.getElementById('modalOverview').innerText = 'Fetching movie details...';
      document.getElementById('modalPoster').src = '';
      document.getElementById('modalBackdrop').style.backgroundImage = 'none';

      document.getElementById('tfidfGrid').innerHTML = '<div class="spinner"></div>';
      document.getElementById('genreGrid').innerHTML = '<div class="spinner"></div>';

      try {
        const detailsRes = await fetch(`/movie/id/${tmdbId}`);
        const details = await detailsRes.json();

        document.getElementById('modalTitle').innerText = details.title;
        document.getElementById('modalMeta').innerText = `Release: ${details.release_date || 'N/A'}`;
        document.getElementById('modalOverview').innerText = details.overview || 'No overview available.';
        if (details.poster_url) document.getElementById('modalPoster').src = details.poster_url;
        if (details.backdrop_url) document.getElementById('modalBackdrop').style.backgroundImage = `url('${details.backdrop_url}')`;

        document.getElementById('modalGenres').innerHTML = (details.genres || []).map(g => `<span class="genre-tag">${g.name}</span>`).join('');

        // Fetch Bundle Recommendations
        const bundleRes = await fetch(`/movie/search?query=${encodeURIComponent(details.title)}&tfidf_top_n=12&genre_limit=12`);
        if (bundleRes.ok) {
          const bundle = await bundleRes.json();
          const tfidfCards = (bundle.tfidf_recommendations || []).map(x => x.tmdb).filter(Boolean);
          renderCards(document.getElementById('tfidfGrid'), tfidfCards);
          renderCards(document.getElementById('genreGrid'), bundle.genre_recommendations || []);
        } else {
          document.getElementById('tfidfGrid').innerHTML = '<div style="color:var(--text-muted); grid-column:1/-1;">No TF-IDF similarity match found.</div>';
          const genreRes = await fetch(`/recommend/genre?tmdb_id=${tmdbId}&limit=12`);
          const genreCards = await genreRes.json();
          renderCards(document.getElementById('genreGrid'), genreCards);
        }
      } catch (err) { console.error(err); }
    }

    function closeModal() {
      document.getElementById('movieModal').style.display = 'none';
    }

    function escapeHtml(str) {
      return (str || '').replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
    }
  </script>
</body>
</html>
"""

# =========================
# ROUTES
# =========================
@app.get("/", response_class=HTMLResponse)
def index():
    index_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content=INDEX_HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- HOME FEED (TMDB) ----------
@app.get("/home", response_model=List[TMDBMovieCard])
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50),
):
    """
    Home feed for Streamlit (posters).
    category:
      - trending (trending/movie/day)
      - popular, top_rated, upcoming, now_playing  (movie/{category})
    """
    try:
        if category == "trending":
            data = await tmdb_get("/trending/movie/day", {"language": "en-US"})
            return await tmdb_cards_from_results(data.get("results", []), limit=limit)

        if category not in {"popular", "top_rated", "upcoming", "now_playing"}:
            raise HTTPException(status_code=400, detail="Invalid category")

        data = await tmdb_get(f"/movie/{category}", {"language": "en-US", "page": 1})
        return await tmdb_cards_from_results(data.get("results", []), limit=limit)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Home route failed: {e}")


# ---------- TMDB KEYWORD SEARCH (MULTIPLE RESULTS) ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10),
):
    """
    Returns RAW TMDB shape with 'results' list.
    Streamlit will use it for:
      - dropdown suggestions
      - grid results
    """
    return await tmdb_search_movies(query=query, page=page)


# ---------- MOVIE DETAILS (SAFE ROUTE) ----------
@app.get("/movie/id/{tmdb_id}", response_model=TMDBMovieDetails)
async def movie_details_route(tmdb_id: int):
    return await tmdb_movie_details(tmdb_id)


# ---------- GENRE RECOMMENDATIONS ----------
@app.get("/recommend/genre", response_model=List[TMDBMovieCard])
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50),
):
    """
    Given a TMDB movie ID:
    - fetch details
    - pick first genre
    - discover movies in that genre (popular)
    """
    details = await tmdb_movie_details(tmdb_id)
    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]
    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1,
        },
    )
    cards = await tmdb_cards_from_results(discover.get("results", []), limit=limit)
    return [c for c in cards if c.tmdb_id != tmdb_id]


# ---------- TF-IDF ONLY (debug/useful) ----------
@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50),
):
    recs = tfidf_recommend_titles(title, top_n=top_n)
    return [{"title": t, "score": s} for t, s in recs]


# ---------- BUNDLE: Details + TF-IDF recs + Genre recs ----------
@app.get("/movie/search", response_model=SearchBundleResponse)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30),
):
    """
    This endpoint is for when you have a selected movie and want:
      - movie details
      - TF-IDF recommendations (local) + posters
      - Genre recommendations (TMDB) + posters

    NOTE:
    - It selects the BEST match from TMDB for the given query.
    - If you want MULTIPLE matches, use /tmdb/search
    """
    best = await tmdb_search_first(query)
    if not best:
        raise HTTPException(
            status_code=404, detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])
    details = await tmdb_movie_details(tmdb_id)

    # 1) TF-IDF recommendations (never crash endpoint)
    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []
    try:
        # try local dataset by TMDB title
        recs = tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
    except Exception:
        # fallback to user query
        try:
            recs = tfidf_recommend_titles(query, top_n=tfidf_top_n)
        except Exception:
            recs = []

    for title, score in recs:
        card = await attach_tmdb_card_by_title(title)
        tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))

    # 2) Genre recommendations (TMDB discover by first genre)
    genre_recs: List[TMDBMovieCard] = []
    if details.genres:
        genre_id = details.genres[0]["id"]
        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1,
            },
        )
        cards = await tmdb_cards_from_results(
            discover.get("results", []), limit=genre_limit
        )
        genre_recs = [c for c in cards if c.tmdb_id != details.tmdb_id]

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs,
    )