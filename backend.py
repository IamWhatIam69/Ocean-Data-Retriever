import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import openai
from argopy import DataFetcher

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI app
app = FastAPI()

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body
class Query(BaseModel):
    query: str


# --- Step 1: Parse query into filters using LLM ---
def parse_query_to_filters(q: str):
    parser_prompt = [
        {"role": "system", "content": "Extract filters for Argo float data queries. Always return JSON."},
        {"role": "user", "content": f"User asked: '{q}'. "
         "Return JSON with keys: region (array of 4 floats [lon_min, lon_max, lat_min, lat_max]), "
         "depth_min (float), depth_max (float), "
         "date_min (YYYY-MM), date_max (YYYY-MM)."}
    ]

    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=parser_prompt,
        temperature=0
    )
    parsed_text = resp["choices"][0]["message"]["content"]

    try:
        filters = json.loads(parsed_text)
    except Exception:
        # Default fallback
        filters = {
            "region": [-60, -30, 20, 40],
            "depth_min": 0,
            "depth_max": 2000,
            "date_min": "2023-01",
            "date_max": "2023-12"
        }
    return filters


# --- Step 2: Fetch Argo data (using ERDDAP backend) ---
def fetch_argo_data(region, depth_min=0, depth_max=2000,
                    date_min="2023-01", date_max="2023-12", limit=10):
    try:
        argo = DataFetcher(src="erddap").region([*region, depth_min, depth_max, date_min, date_max])
        ds = argo.to_xarray().isel(N_PROF=slice(0, limit))
        df = ds.to_dataframe().reset_index()
    except Exception as e:
        print("⚠️ Argo fetch failed:", e)
        return [], []

    summaries, viz = [], []
    for _, row in df.iterrows():
        try:
            text = (
                f"Float {row['N_PROF']} on {row['JULD']} at "
                f"{row['LATITUDE']}N, {row['LONGITUDE']}E "
                f"depth {row['PRES']}m: T={row['TEMP']}°C, S={row['PSAL']} PSU"
            )
            summaries.append(text)
            viz.append({
                "lat": row["LATITUDE"],
                "lon": row["LONGITUDE"],
                "depth": row["PRES"],
                "temp": row["TEMP"],
                "salinity": row["PSAL"]
            })
        except Exception:
            continue

    return summaries, viz


# --- Step 3: Main API endpoint ---
@app.post("/ask")
def ask_backend(item: Query):
    q = item.query

    # Parse filters from query
    filters = parse_query_to_filters(q)

    # Fetch matching data
    data_texts, viz_data = fetch_argo_data(
        region=filters["region"],
        depth_min=filters["depth_min"],
        depth_max=filters["depth_max"],
        date_min=filters["date_min"],
        date_max=filters["date_max"]
    )

    if not data_texts:
        return {"answer": "No Argo data found for your request.", "visualization_data": []}

    # Summarize with LLM
    llm_prompt = [
        {"role": "system", "content": "You are an oceanography expert."},
        {"role": "user", "content": f"Answer the query: {q}. Here is the data: {data_texts}"}
    ]
    answer = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=llm_prompt,
        temperature=0.5
    )["choices"][0]["message"]["content"]

    return {"answer": answer, "visualization_data": viz_data}
