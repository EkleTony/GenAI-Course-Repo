from fastapi import FastAPI
import polars as pl
from sentence_transformers import SentenceTransformer  # text embedding
from sklearn.metrics import DistanceMetric  # distance queury

from app.functions import return_search_result_indexes


# Load embedding model
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Load video/document index
df = pl.scan_parquet("app/data/video-index.parquet")

# Distance metric
dist = DistanceMetric.get_metric("manhattan")


# create FastAPI app
app = FastAPI()

# API operataions
# 1...GET-- allows user to send requrest to api and get request


@app.get("/")  # root endpoint
def health_check():
    return {"health_check": "OK"}

# others PuT, Post and Delete


@app.get("/info")  # end point info about the query
def info():
    return {
        "name": "yt-search-api",
        "description": "Semantic search API using FastAPI and sentence embeddings",
    }


@app.get("/search")
def search(query: str):
    result_indexes = return_search_result_indexes(query, df, model, dist)

    results = (
        df.select(["title", "video_id"])
        .collect()
        [result_indexes]
        .to_dict(as_series=False)
    )

    return {
        "query": query,
        "results": results,
    }
