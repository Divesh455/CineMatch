# 🎬 CineMatch - AI Movie Recommender System

A high-performance, content-based Movie Recommendation System built with **FastAPI**, **Scikit-Learn**, **Pandas**, and a modern embedded **Single-Page Web UI**.

It combines Machine Learning **TF-IDF Content-Based Filtering** on local movie datasets with real-time metadata & imagery powered by **The Movie Database (TMDB) API**.

---

## ✨ Features

- **Embedded Glassmorphism Web UI**: Fast, responsive single-page web app served directly from FastAPI at `http://127.0.0.1:8000/`.
- **TF-IDF Machine Learning Recommendations**: Calculates cosine similarity on movie overviews, genres, cast, and crew metadata.
- **TMDB Live Search & Autocomplete**: Real-time keyword search suggestions with titles and release dates.
- **Category Feeds**: Quick filter pills for **Popular**, **Trending Today**, **Top Rated**, and **Upcoming** movies.
- **Interactive Details Modal**: Displays backdrop heroes, poster images, full plot overviews, genre tags, and recommendation grids.

---

## 🌐 Netlify Deployment Guide

This project is configured for **Netlify Serverless Deployment** using `mangum` (FastAPI AWS Lambda/Netlify adapter) and `netlify.toml`.

### Step-by-Step Netlify Deployment:

1. **Push Code to GitHub / GitLab / Bitbucket**:
   Ensure `netlify.toml`, `netlify/functions/api.py`, `requirements.txt`, `main.py`, and `models/` folder are committed to your repository.

2. **Connect Repository to Netlify**:
   - Log into [Netlify](https://app.netlify.com/).
   - Click **Add new site** → **Import an existing project**.
   - Select your Git repository branch.

3. **Configure Build Settings**:
   - **Base directory**: `Reccomender UI` (or leave empty if repository root is `Reccomender UI`)
   - **Build command**: (leave blank)
   - **Publish directory**: (leave blank)

4. **Add Environment Variables**:
   In Netlify Site Settings → **Environment variables**:
   - `TMDB_API_KEY`: `your_tmdb_api_key_here`

5. **Deploy Site**:
   Click **Deploy Site**. Netlify will build the serverless Python function and serve your complete FastAPI app and UI live on `.netlify.app`.

---

## 🧠 Model Training & Similarity Pipeline

The recommendation engine relies on **Content-Based Filtering** using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity**.

### 1. Data Cleaning & Feature Engineering
- Dataset source: `TMDB & IMDB Movies Dataset` containing over 30,000+ movies.
- Text fields selected for feature extraction:
  - `title`: Movie name
  - `overview`: Plot summary
  - `genres`: Action, Sci-Fi, Drama, etc.
  - `cast`: Main actors/actresses
  - `directors`: Film directors
  - `production_companies`: Studios
- Text fields are normalized (lowercased, punctuation removed) and concatenated into a unified metadata string per movie.

### 2. Vectorization & Similarity Matrix Computation
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_features=50000, ngram_range=(1, 2))

# Fit and transform metadata corpus into sparse TF-IDF matrix
tfidf_matrix = tfidf.fit_transform(movie_info['tags'])

# Compute index lookup mapping (normalized title -> row index)
indices = pd.Series(movie_info.index, index=movie_info['title']).drop_duplicates()
```

### 3. Model Serialization & Storage
The trained artifacts are serialized using `pickle` into the `models/` directory:
- `movie_info.pkl`: Filtered DataFrame containing movie IDs, titles, overviews, and posters.
- `indices.pkl`: Pandas Series mapping normalized movie titles to matrix row indices.
- `tfidf_matrix.pkl`: Precomputed sparse SciPy `csr_matrix` storing TF-IDF vector representations (31,446 movies × 50,000 features).
- `tfidf.pkl`: Trained `TfidfVectorizer` object.

### 4. Recommendation Inference Logic
When a user searches for recommendations for a given title:
1. Lookup the row index $i$ of the query title in `indices.pkl`.
2. Retrieve vector $v_i = \text{tfidf\_matrix}[i]$.
3. Compute cosine similarity scores: $S = \text{tfidf\_matrix} \cdot v_i^T$.
4. Sort scores descending and return the top-N matches (excluding the query movie itself).

---

## 📁 Project Structure

```text
Reccomender UI/
├── models/                     # Trained ML Model Artifacts
│   ├── movie_info.pkl
│   ├── indices.pkl
│   ├── tfidf_matrix.pkl
│   └── tfidf.pkl
├── netlify/
│   └── functions/
│       └── api.py              # Netlify Serverless Function Handler
├── netlify.toml                # Netlify Configuration & Redirects
├── main.py                     # FastAPI Backend & Embedded Web UI
├── .env                        # Environment Variables
├── requirements.txt            # Python Dependencies (includes mangum)
└── README.md                   # Project Documentation
```

---

## ⚙️ Local Setup & Installation

### 1. Prerequisites
- Python 3.10+

### 2. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 3. Configure `.env` File
Create or verify `.env` in `Reccomender UI`:
```env
TMDB_API_KEY="YOUR_TMDB_API_KEY_HERE"
API_BASE="http://127.0.0.1:8000"
```

---

## 🚀 Running Locally

Start the FastAPI server:

```powershell
uvicorn main:app --reload
```

Then open your browser and navigate to:
👉 **`http://127.0.0.1:8000/`**
