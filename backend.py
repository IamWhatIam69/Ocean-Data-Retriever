from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from argopy import DataFetcher
import re

# --- Setup FastAPI ---
app = FastAPI()

# Allow frontend (your HTML) to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    visualization_data: dict | None = None

# --- Helper: parse query ---
def parse_query(query: str):
    """Very basic keyword parser for demo."""
    if "temperature" in query.lower():
        return "temperature"
    elif "salinity" in query.lower():
        return "salinity"
    elif "float" in query.lower():
        return "floats"
    return "general"

# --- API Endpoint ---
@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    query = request.query
    intent = parse_query(query)

    try:
        if intent == "temperature":
            # Example: Fetch global Argo temperature profiles
            ds = DataFetcher().profile(selection=dict(lat=[-10,10], lon=[120,160], depth=[0,10])).to_xarray()
            temp = float(ds['TEMP'].mean())
            answer = f"Average sea temperature in the region is about {temp:.2f} °C."
            return QueryResponse(answer=answer)

        elif intent == "salinity":
            ds = DataFetcher().profile(selection=dict(lat=[-10,10], lon=[120,160], depth=[0,10])).to_xarray()
            sal = float(ds['PSAL'].mean())
            answer = f"Average salinity in the region is about {sal:.2f} PSU."
            return QueryResponse(answer=answer)

        elif intent == "floats":
            # Fetch latest float positions
            df = DataFetcher().float([6902746, 6902914]).to_dataframe()
            floats = df.groupby("N_PROF").first().reset_index()
            markers = [
                {"lat": float(row["LATITUDE"]), "lon": float(row["LONGITUDE"]), "id": int(row["PLATFORM_NUMBER"])}
                for _, row in floats.iterrows()
            ]
            return QueryResponse(
                answer=f"I found {len(markers)} Argo floats in the dataset.",
                visualization_data={"markers": markers}
            )

        else:
            return QueryResponse(answer="I can help you with ocean temperature, salinity, or float locations. Try asking me one of those!")

    except Exception as e:
        return QueryResponse(answer=f"Error while fetching data: {str(e)}")

# --- Run Server ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)