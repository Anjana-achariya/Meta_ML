from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid

from recommendation.recommend_model import recommend_models

app = FastAPI(
    title="Meta-Learning Model Recommender",
    version="1.0"
)

# ---------------- Directories ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------- Home Page ----------------
@app.get("/", response_class=HTMLResponse)
async def home():
    try:
        with open(os.path.join(STATIC_DIR, "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h2>UI not found</h2>"


# ---------------- Recommendation Endpoint ----------------
@app.post("/recommend")
async def recommend(file: UploadFile = File(...)):
    try:
        # Unique filename to avoid collisions
        unique_name = f"{uuid.uuid4()}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, unique_name)

        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Call recommender (default weights inside function)
        result = recommend_models(file_path)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
