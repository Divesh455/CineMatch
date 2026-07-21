import sys
import os

# Add parent directory to sys.path to import main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from main import app, load_pickles
from mangum import Mangum

# Trigger model loading on function initialization
try:
    load_pickles()
except Exception as e:
    print(f"Error loading pickles in Netlify serverless function: {e}")

handler = Mangum(app)
