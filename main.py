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
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #06080e;
      --surface: rgba(16, 20, 34, 0.7);
      --surface-hover: rgba(26, 33, 56, 0.95);
      --accent: #a855f7;
      --accent-gradient: linear-gradient(135deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --border: rgba(255, 255, 255, 0.08);
      --border-glowing: rgba(168, 85, 247, 0.4);
      --modal-bg: rgba(9, 12, 22, 0.97);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    
    body {
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
      background-image: 
        radial-gradient(circle at 12% 10%, rgba(99, 102, 241, 0.15) 0%, transparent 45%),
        radial-gradient(circle at 88% 55%, rgba(168, 85, 247, 0.12) 0%, transparent 50%),
        radial-gradient(circle at 50% 90%, rgba(236, 72, 153, 0.08) 0%, transparent 45%);
      background-attachment: fixed;
    }

    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.15); border-radius: 9999px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(168, 85, 247, 0.5); }

    /* Glassmorphism Header */
    header {
      position: sticky; top: 0; z-index: 100;
      background: rgba(6, 8, 14, 0.85); backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
      border-bottom: 1px solid var(--border); padding: 1.15rem 3rem;
      display: grid; grid-template-columns: 200px 1fr 200px; align-items: center; gap: 1.5rem;
    }

    .logo {
      font-family: 'Outfit', sans-serif;
      font-size: 1.75rem; font-weight: 800; letter-spacing: -0.5px;
      background: var(--accent-gradient); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
      display: flex; align-items: center; gap: 0.6rem; cursor: pointer; user-select: none;
      filter: drop-shadow(0 2px 10px rgba(168, 85, 247, 0.35));
      transition: transform 0.25s ease;
    }
    .logo:hover { transform: scale(1.03); }

    .search-box { position: relative; width: 100%; max-width: 640px; margin: 0 auto; }
    .search-input {
      width: 100%; padding: 0.85rem 1.6rem; border-radius: 9999px;
      background: rgba(255,255,255,0.035); border: 1px solid var(--border);
      color: var(--text); font-size: 0.95rem; outline: none; transition: all 0.3s ease;
      box-shadow: inset 0 2px 4px rgba(0,0,0,0.3); text-align: center;
    }
    .search-input::placeholder { text-align: center; }
    .search-input:focus {
      border-color: rgba(168, 85, 247, 0.6);
      box-shadow: 0 0 30px rgba(168, 85, 247, 0.28), inset 0 2px 4px rgba(0,0,0,0.2);
      background: rgba(255,255,255,0.06); text-align: left;
    }
    .search-input:focus::placeholder { text-align: left; }

    /* Search Suggestions with Mini Posters */
    .suggestions {
      position: absolute; top: 118%; left: 0; right: 0; background: #0c101c;
      border: 1px solid var(--border-glowing); border-radius: 20px; max-height: 420px; overflow-y: auto;
      box-shadow: 0 25px 60px -10px rgba(0,0,0,0.95); z-index: 200; display: none; padding: 0.5rem;
    }
    .suggestion-item {
      padding: 0.65rem 0.95rem; border-radius: 14px; cursor: pointer; transition: all 0.2s ease;
      display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255,255,255,0.03);
      gap: 1rem;
    }
    .suggestion-item:last-child { border-bottom: none; }
    .suggestion-item:hover { background: rgba(168, 85, 247, 0.2); transform: translateX(3px); }
    .suggestion-left { display: flex; align-items: center; gap: 0.9rem; flex: 1; min-width: 0; }
    .suggestion-thumb { width: 40px; height: 58px; object-fit: cover; border-radius: 8px; background: #131826; flex-shrink: 0; border: 1px solid var(--border); }
    .suggestion-thumb-placeholder { width: 40px; height: 58px; border-radius: 8px; background: #131826; display: flex; align-items: center; justify-content: center; font-size: 1.1rem; flex-shrink: 0; border: 1px solid var(--border); }
    .suggestion-title { font-family: 'Outfit', sans-serif; font-size: 0.96rem; font-weight: 600; color: #fff; line-height: 1.25; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .suggestion-meta { font-size: 0.82rem; color: var(--text-muted); margin-top: 2px; }
    .suggestion-rating { font-size: 0.78rem; font-weight: 700; color: #fbbf24; background: rgba(6,8,14,0.7); padding: 3px 8px; border-radius: 8px; flex-shrink: 0; border: 1px solid rgba(251, 191, 36, 0.25); }

    main { max-width: 1440px; margin: 0 auto; padding: 2.2rem 3rem; }

    /* Top Middle Featured Spotlight Movie Banner */
    .featured-banner {
      position: relative; width: 100%; height: 350px; border-radius: 28px; overflow: hidden;
      margin-bottom: 2.5rem; border: 1px solid var(--border-glowing);
      box-shadow: 0 25px 50px -15px rgba(0,0,0,0.85); display: flex; align-items: flex-end;
    }
    .featured-backdrop {
      position: absolute; inset: 0; background-size: cover; background-position: center;
      transition: background-image 0.5s ease-in-out; filter: brightness(0.85);
    }
    .featured-overlay {
      position: absolute; inset: 0;
      background: linear-gradient(to right, rgba(6, 8, 14, 0.95) 0%, rgba(6, 8, 14, 0.75) 45%, rgba(6, 8, 14, 0.3) 100%),
                  linear-gradient(to top, rgba(6, 8, 14, 0.95) 0%, transparent 60%);
    }
    .featured-content {
      position: relative; z-index: 10; padding: 2.5rem; max-width: 700px;
      display: flex; flex-direction: column; gap: 0.7rem;
    }
    .featured-badge {
      align-self: flex-start; background: var(--accent-gradient); color: #fff;
      padding: 4px 14px; border-radius: 9999px; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.5px;
      box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }
    .featured-title {
      font-family: 'Outfit', sans-serif; font-size: 2.3rem; font-weight: 800; color: #fff; line-height: 1.15;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .featured-meta { font-size: 0.9rem; color: #e2e8f0; font-weight: 600; }
    .featured-overview {
      color: var(--text-muted); font-size: 0.95rem; line-height: 1.6;
      display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
    }
    .featured-btn {
      align-self: flex-start; margin-top: 0.4rem; padding: 0.75rem 1.6rem; border-radius: 9999px;
      background: #fff; color: #07090e; border: none; font-weight: 700; font-size: 0.92rem;
      cursor: pointer; transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1); box-shadow: 0 10px 25px rgba(255,255,255,0.25);
    }
    .featured-btn:hover { transform: translateY(-3px) scale(1.03); background: #f1f5f9; box-shadow: 0 14px 30px rgba(255,255,255,0.35); }

    /* Category Nav Pills */
    .nav-pills {
      display: flex; justify-content: center; align-items: center; gap: 0.85rem;
      margin-bottom: 2.2rem; overflow-x: auto; padding-bottom: 0.5rem; scrollbar-width: none; flex-wrap: wrap;
    }
    .nav-pills::-webkit-scrollbar { display: none; }
    .pill {
      padding: 0.75rem 1.6rem; border-radius: 9999px; background: rgba(255,255,255,0.035);
      border: 1px solid var(--border); color: var(--text-muted); cursor: pointer;
      font-weight: 600; font-size: 0.92rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); white-space: nowrap;
    }
    .pill:hover { background: rgba(255,255,255,0.08); color: var(--text); border-color: rgba(255,255,255,0.2); transform: translateY(-2px); }
    .pill.active {
      background: var(--accent-gradient); color: #fff; border-color: transparent;
      box-shadow: 0 8px 25px rgba(168, 85, 247, 0.45); transform: translateY(-2px);
    }

    .section-title { font-family: 'Outfit', sans-serif; font-size: 1.55rem; font-weight: 700; letter-spacing: -0.4px; margin-bottom: 1.6rem; color: #fff; text-align: center; }

    /* Movie Cards Grid */
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 1.8rem; }

    .card {
      background: var(--surface); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
      border-radius: 22px; overflow: hidden;
      border: 1px solid var(--border); cursor: pointer; transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
      display: flex; flex-direction: column; position: relative;
    }
    .card:hover {
      transform: translateY(-10px) scale(1.025);
      background: var(--surface-hover);
      border-color: var(--border-glowing);
      box-shadow: 0 24px 48px -12px rgba(168, 85, 247, 0.32);
    }
    .card-img-wrap { width: 100%; aspect-ratio: 2/3; background: #0c101c; position: relative; overflow: hidden; }
    .card-img { width: 100%; height: 100%; object-fit: cover; transition: transform 0.5s cubic-bezier(0.16, 1, 0.3, 1); }
    .card:hover .card-img { transform: scale(1.09); }
    .no-img { display: flex; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 2.2rem; }

    .rating-badge {
      position: absolute; top: 12px; right: 12px; background: rgba(6, 8, 14, 0.85);
      backdrop-filter: blur(12px); padding: 5px 10px; border-radius: 10px; font-size: 0.8rem; font-weight: 700; color: #fbbf24;
      border: 1px solid rgba(251, 191, 36, 0.35); box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    }

    .card-info { padding: 1.15rem; display: flex; flex-direction: column; gap: 0.4rem; flex: 1; }
    .card-title { font-family: 'Outfit', sans-serif; font-size: 1.02rem; font-weight: 700; line-height: 1.3; color: #fff; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
    .card-meta { font-size: 0.84rem; color: var(--text-muted); }

    /* Shimmer Skeleton Loaders */
    .skeleton-card {
      background: var(--surface); border-radius: 22px; border: 1px solid var(--border);
      aspect-ratio: 2/3.4; position: relative; overflow: hidden;
    }
    .skeleton-card::after {
      content: ""; position: absolute; inset: 0;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
      animation: shimmer 1.5s infinite;
    }
    @keyframes shimmer { 0% { transform: translateX(-100%); } 100% { transform: translateX(100%); } }

    /* Modal */
    .modal {
      position: fixed; inset: 0; z-index: 300; background: rgba(0,0,0,0.88); backdrop-filter: blur(22px);
      display: none; align-items: center; justify-content: center; padding: 2rem; overflow-y: auto;
      animation: fadeIn 0.25s ease-out;
    }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

    .modal-content {
      background: var(--modal-bg); border: 1px solid var(--border-glowing); border-radius: 28px;
      max-width: 1080px; width: 100%; max-height: 90vh; overflow-y: auto; position: relative;
      box-shadow: 0 35px 75px -15px rgba(0,0,0,0.95); scrollbar-width: thin;
      animation: modalSlideUp 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    }
    @keyframes modalSlideUp { from { transform: translateY(20px) scale(0.97); } to { transform: translateY(0) scale(1); } }

    .modal-close {
      position: absolute; top: 22px; right: 24px; z-index: 10; background: rgba(255,255,255,0.08);
      border: 1px solid var(--border); color: #fff; width: 42px; height: 42px; border-radius: 50%; cursor: pointer;
      font-size: 1.2rem; transition: all 0.25s; display: flex; align-items: center; justify-content: center;
    }
    .modal-close:hover { background: rgba(239, 68, 68, 0.85); transform: rotate(90deg); }

    .backdrop-container { width: 100%; height: 320px; position: relative; background-size: cover; background-position: center; border-bottom: 1px solid var(--border); }
    .backdrop-overlay { position: absolute; inset: 0; background: linear-gradient(to bottom, transparent 15%, var(--modal-bg) 100%); }

    .details-header { display: flex; gap: 2.4rem; padding: 0 2.8rem 2.2rem 2.8rem; margin-top: -100px; position: relative; z-index: 2; flex-wrap: wrap; }
    .poster-large { width: 200px; border-radius: 20px; border: 2px solid var(--border-glowing); box-shadow: 0 25px 50px rgba(0,0,0,0.85); object-fit: cover; aspect-ratio: 2/3; background: #0c101c; flex-shrink: 0; }

    .details-body { flex: 1; min-width: 280px; display: flex; flex-direction: column; gap: 0.85rem; justify-content: flex-end; }
    .details-title { font-family: 'Outfit', sans-serif; font-size: 2.5rem; font-weight: 800; color: #fff; line-height: 1.15; }
    .genres-list { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 4px; }
    .genre-tag { background: rgba(168, 85, 247, 0.18); border: 1px solid rgba(168, 85, 247, 0.4); color: #c084fc; padding: 4px 14px; border-radius: 9999px; font-size: 0.84rem; font-weight: 600; }
    .overview-text { color: var(--text-muted); line-height: 1.8; font-size: 1.02rem; margin-top: 0.5rem; }

    .trailer-btn {
      align-self: flex-start; margin-top: 0.6rem; padding: 0.75rem 1.6rem; border-radius: 9999px;
      background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); color: #fff;
      text-decoration: none; font-weight: 700; font-size: 0.92rem; display: inline-flex;
      align-items: center; gap: 0.6rem; transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
      box-shadow: 0 8px 20px rgba(239, 68, 68, 0.4); border: none; cursor: pointer;
    }
    .trailer-btn:hover {
      transform: translateY(-2px) scale(1.03);
      box-shadow: 0 12px 28px rgba(239, 68, 68, 0.55); background: linear-gradient(135deg, #f87171 0%, #ef4444 100%);
    }

    .recs-section { padding: 2.4rem 2.8rem 2.8rem 2.8rem; border-top: 1px solid var(--border); display: flex; flex-direction: column; gap: 2.5rem; }

    @media (max-width: 768px) {
      header { padding: 1rem 1.5rem; }
      main { padding: 1.5rem; }
      .featured-banner { height: 380px; }
      .featured-content { padding: 1.5rem; }
      .featured-title { font-size: 1.8rem; }
      .details-header { margin-top: -60px; padding: 0 1.5rem 1.5rem 1.5rem; }
      .poster-large { width: 145px; }
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

    <div style="width: 200px;"></div>
  </header>

  <main>
    <!-- Top Middle Featured Trending Movie Banner -->
    <div class="featured-banner" id="featuredBanner" style="display:none;">
      <div class="featured-backdrop" id="featuredBackdrop"></div>
      <div class="featured-overlay"></div>
      <div class="featured-content">
        <div class="featured-badge" id="featuredBadge">🔥 Featured Spotlight</div>
        <h1 class="featured-title" id="featuredTitle">Movie Title</h1>
        <div class="featured-meta" id="featuredMeta">Release Date</div>
        <p class="featured-overview" id="featuredOverview">Overview...</p>
        <button class="featured-btn" id="featuredBtn">▶ Explore Details</button>
      </div>
    </div>

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
          <a class="trailer-btn" id="modalTrailerBtn" href="#" target="_blank" rel="noopener noreferrer">
            ▶ Watch Trailer
          </a>
        </div>
      </div>

      <div class="recs-section">
        <div>
          <div class="section-title" style="text-align: left; margin-bottom: 1.2rem;">🔎 Similar Movies (TF-IDF Similarity)</div>
          <div class="grid" id="tfidfGrid"></div>
        </div>

        <div>
          <div class="section-title" style="text-align: left; margin-bottom: 1.2rem;">🎭 More Like This (Genre Match)</div>
          <div class="grid" id="genreGrid"></div>
        </div>
      </div>

    </div>
  </div>

  <script>
    const API_BASE = window.location.origin.includes('netlify.app') ? '/api' : '';
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

    function renderSkeletons(container, count = 12) {
      container.innerHTML = Array(count).fill('<div class="skeleton-card"></div>').join('');
    }

    function updateSpotlight(movie, categoryLabel) {
      if (!movie) return;
      const banner = document.getElementById('featuredBanner');
      const backdrop = document.getElementById('featuredBackdrop');
      const title = document.getElementById('featuredTitle');
      const meta = document.getElementById('featuredMeta');
      const overview = document.getElementById('featuredOverview');
      const btn = document.getElementById('featuredBtn');
      const badge = document.getElementById('featuredBadge');

      badge.innerText = `✨ Featured Spotlight (${categoryLabel})`;
      title.innerText = movie.title || '';
      meta.innerText = `${movie.vote_average ? '⭐ ' + movie.vote_average.toFixed(1) + ' • ' : ''}${(movie.release_date || '').slice(0,4)}`;
      overview.innerText = movie.overview || 'Explore details and recommendations for this featured title.';
      
      const imgPath = movie.backdrop_url || movie.poster_url;
      if (imgPath) {
        backdrop.style.backgroundImage = `url('${imgPath}')`;
      }

      btn.onclick = () => openMovieDetails(movie.tmdb_id, movie.title);
      banner.style.display = 'flex';
    }

    async function loadCategory(cat, btn) {
      if (btn) {
        document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
      }

      const grid = document.getElementById('movieGrid');
      const title = document.getElementById('gridTitle');
      const catLabel = btn ? btn.innerText.replace(/^[^\s]+\s*/, '') : 'Popular';
      title.innerText = btn ? btn.innerText + ' Movies' : 'Movies';
      renderSkeletons(grid, 12);

      try {
        const res = await fetch(`${API_BASE}/home?category=${cat}&limit=24`);
        const data = await res.json();
        if (data && data.length > 0) {
          updateSpotlight(data[0], catLabel);
        }
        renderCards(grid, data);
      } catch (err) {
        grid.innerHTML = `<div style="color:#ef4444; grid-column:1/-1;">Failed to load movies: ${err}</div>`;
      }
    }

    async function fetchSuggestions(query) {
      try {
        const res = await fetch(`${API_BASE}/tmdb/search?query=${encodeURIComponent(query)}`);
        const data = await res.json();
        const results = data.results || [];
        const box = document.getElementById('suggestionsBox');

        if (!results.length) {
          box.style.display = 'none';
          return;
        }

        box.innerHTML = results.slice(0, 8).map(m => {
          const posterUrl = m.poster_path ? `https://image.tmdb.org/t/p/w92${m.poster_path}` : null;
          return `
            <div class="suggestion-item" onclick="openMovieDetails(${m.id}, '${escapeHtml(m.title)}')">
              <div class="suggestion-left">
                ${posterUrl ? `<img class="suggestion-thumb" src="${posterUrl}" alt="${escapeHtml(m.title)}">` : `<div class="suggestion-thumb-placeholder">🎬</div>`}
                <div>
                  <div class="suggestion-title">${escapeHtml(m.title)}</div>
                  <div class="suggestion-meta">${(m.release_date || '').slice(0,4)}</div>
                </div>
              </div>
              ${m.vote_average ? `<span class="suggestion-rating">⭐ ${m.vote_average.toFixed(1)}</span>` : ''}
            </div>
          `;
        }).join('');
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
      const modalContent = document.querySelector('.modal-content');
      if (modalContent) modalContent.scrollTop = 0;

      modal.style.display = 'flex';

      document.getElementById('modalTitle').innerText = title || 'Loading...';
      document.getElementById('modalMeta').innerText = '';
      document.getElementById('modalGenres').innerHTML = '';
      document.getElementById('modalOverview').innerText = 'Fetching movie details...';
      document.getElementById('modalPoster').src = '';
      document.getElementById('modalBackdrop').style.backgroundImage = 'none';

      const trailerBtn = document.getElementById('modalTrailerBtn');
      trailerBtn.href = `https://www.youtube.com/results?search_query=${encodeURIComponent((title || '') + ' official trailer')}`;

      renderSkeletons(document.getElementById('tfidfGrid'), 6);
      renderSkeletons(document.getElementById('genreGrid'), 6);

      try {
        const detailsRes = await fetch(`${API_BASE}/movie/id/${tmdbId}`);
        const details = await detailsRes.json();

        document.getElementById('modalTitle').innerText = details.title;
        document.getElementById('modalMeta').innerText = `Release: ${details.release_date || 'N/A'}`;
        document.getElementById('modalOverview').innerText = details.overview || 'No overview available.';
        if (details.poster_url) document.getElementById('modalPoster').src = details.poster_url;
        if (details.backdrop_url) document.getElementById('modalBackdrop').style.backgroundImage = `url('${details.backdrop_url}')`;

        document.getElementById('modalGenres').innerHTML = (details.genres || []).map(g => `<span class="genre-tag">${g.name}</span>`).join('');
        trailerBtn.href = `https://www.youtube.com/results?search_query=${encodeURIComponent(details.title + ' official trailer')}`;

        // Fetch Bundle Recommendations
        const bundleRes = await fetch(`${API_BASE}/movie/search?query=${encodeURIComponent(details.title)}&tfidf_top_n=12&genre_limit=12`);
        if (bundleRes.ok) {
          const bundle = await bundleRes.json();
          const tfidfCards = (bundle.tfidf_recommendations || []).map(x => x.tmdb).filter(Boolean);
          renderCards(document.getElementById('tfidfGrid'), tfidfCards);
          renderCards(document.getElementById('genreGrid'), bundle.genre_recommendations || []);
        } else {
          document.getElementById('tfidfGrid').innerHTML = '<div style="color:var(--text-muted); grid-column:1/-1;">No TF-IDF similarity match found.</div>';
          const genreRes = await fetch(`${API_BASE}/recommend/genre?tmdb_id=${tmdbId}&limit=12`);
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


# ---------- TMDB KEYWORD SEARCH (FILTERED TO LOCAL MOVIE_INFO) ----------
@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = 1,
):
    """
    Returns search results filtered strictly to movies present in movie_info.
    """
    global df
    raw = await tmdb_search_movies(query=query, page=page)
    results = raw.get("results", [])
    
    filtered = []
    seen_ids = set()
    
    # Set of IDs in local df for fast lookup if available
    local_df_ids = set()
    if df is not None and "id" in df.columns:
        local_df_ids = set(df["id"].dropna().astype(int).values)
    
    for m in results:
        m_title = m.get("title", "")
        m_id = m.get("id")
        # Include if title maps to local TF-IDF index or ID is in df
        if (get_local_idx_by_title(m_title) is not None) or (m_id and m_id in local_df_ids):
            filtered.append(m)
            if m_id:
                seen_ids.add(m_id)

    # Local substring search in df for dataset movies
    if df is not None:
        q_lower = query.lower()
        local_matches = df[df["title"].astype(str).str.lower().str.contains(q_lower, regex=False, na=False)].head(10)
        for _, row in local_matches.iterrows():
            row_id = int(row["id"]) if "id" in row and pd.notnull(row["id"]) else None
            if row_id and row_id not in seen_ids:
                poster_p = str(row["poster_path"]) if "poster_path" in row and pd.notnull(row["poster_path"]) else None
                backdrop_p = str(row["backdrop_path"]) if "backdrop_path" in row and pd.notnull(row["backdrop_path"]) else None
                filtered.append({
                    "id": row_id,
                    "title": str(row["title"]),
                    "overview": str(row.get("overview", "")),
                    "release_date": "",
                    "poster_path": poster_p,
                    "backdrop_path": backdrop_p,
                    "vote_average": 0.0,
                })
                seen_ids.add(row_id)

    return {"results": filtered, "page": 1, "total_pages": 1, "total_results": len(filtered)}


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