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

## 🚀 Deployment Guide

### Option A: Deploy on Render (Recommended for full Python ML & FastAPI)

Render provides full Python Web Service support with dedicated RAM and Uvicorn server execution, making it ideal for unpickling Scikit-Learn TF-IDF models and running FastAPI seamlessly.

1. **Push code to GitHub / GitLab**.
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** → **Web Service**.
4. Select your repository branch.
5. Configure settings:
   - **Root Directory**: `Reccomender UI` (or leave blank if repository root)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Add Environment Variable:
   - `TMDB_API_KEY`: `your_tmdb_api_key_here`
7. Click **Create Web Service**. Your app will be live on `https://cinematch.onrender.com`!

---

### Option B: Deploy on Netlify (Serverless)

Configured via `netlify.toml` and `mangum` adapter in `netlify/functions/api.py`.

1. Import repository into [Netlify](https://app.netlify.com/).
2. Set Base Directory: `Reccomender UI`.
3. Add Environment Variable: `TMDB_API_KEY`.
4. Deploy site!

---

## 🧠 Model Training & Similarity Pipeline

The recommendation engine relies on **Content-Based Filtering** using **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity**.

### 1. Data Cleaning & Feature Engineering
- Dataset source: `TMDB & IMDB Movies Dataset` containing over 30,000+ movies.
- Text fields selected for feature extraction: `title`, `overview`, `genres`, `cast`, `directors`, `production_companies`.
- Text fields are normalized and concatenated into a unified metadata string per movie.

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

---

## 📁 Project Structure

```text
Reccomender UI/
├── models/                     # Trained ML Model Artifacts
│   ├── movie_info.pkl
│   ├── indices.pkl
│   ├── tfidf_matrix.pkl
│   └── tfidf.pkl
├── Procfile                    # Render Web Service Process Command
├── render.yaml                 # Render Infrastructure-as-Code Configuration
├── main.py                     # FastAPI Backend & Embedded Web UI
├── index.html                  # Single-Page Frontend Application
├── .env                        # Environment Variables
├── requirements.txt            # Python Dependencies
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
