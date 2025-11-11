from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.base import create_sql_agent

from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# === FASTAPI APP INITIALIZATION ===
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === DATABASE & AGENT SETUP ===
load_dotenv()

DB_URI = os.getenv("DB_URI")

db = SQLDatabase.from_uri(DB_URI)
engine = create_engine(DB_URI)

llm = ChatOllama(
    model="llama3.2",
    base_url="http://localhost:11434",
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# ✅ FIX: New valid agent_type value
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type="zero-shot-react-description",
    verbose=True,
)

# === INPUT MODEL FOR CHAT ===
class ChatInput(BaseModel):
    message: str

# === /chat ENDPOINT ===
@app.post("/chat")
async def chat(input: ChatInput):
    response = await run_in_threadpool(agent_executor.run, input.message)
    return {"response": response, "source": "llm"}

# === /forecast ENDPOINT ===
@app.post("/forecast")
async def forecast_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")
    response = await run_in_threadpool(agent_executor.run, message)
    return {"response": response, "source": "llm"}

# === /metrics/{store}/{year} ENDPOINT ===
@app.get("/metrics/{store}/{year}")
def get_metrics(store: int, year: int):
    try:
        df = pd.read_sql("SELECT * FROM pred_sales_data_2", engine)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

    # FIX COLUMN NAMES (Postgres lowercases everything)
    df.columns = [col.lower() for col in df.columns]

    # Parse and preprocess
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=["date"], inplace=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter

    # Filter for store & year
    df_filtered = df[(df["store"] == store) & (df["year"] == year)]

    # Monthly data
    monthly_revenue = (
        df_filtered.groupby("month")["actual_weekly_sales"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
        .tolist()
    )
    monthly_projected = (
        df_filtered.groupby("month")["predicted_weekly_sales"]
        .sum()
        .reindex(range(1, 13), fill_value=0)
        .tolist()
    )

    # Quarterly profit & loss
    profit_series = df_filtered["actual_weekly_sales"] - df_filtered["predicted_weekly_sales"]

    quarterly_profit = (
        profit_series.groupby(df_filtered["quarter"])
        .sum()
        .reindex(range(1, 5), fill_value=0)
        .tolist()
    )

    quarterly_loss = (
        profit_series.clip(upper=0).abs()
        .groupby(df_filtered["quarter"])
        .mean()
        .reindex(range(1, 5), fill_value=0)
        .tolist()
    )

    # Summary Metrics
    total_revenue = df_filtered["predicted_weekly_sales"].sum()
    total_cost = total_revenue * 0.65
    profit_loss = total_revenue - total_cost
    offset = df_filtered["actual_weekly_sales"].sum() - df_filtered["predicted_weekly_sales"].sum()

    return {
        "revenue": f"${int(total_revenue // 1000)}K",
        "margin": f"{int(profit_loss)}K" if total_revenue else "0%",
        "opex": f"{int((total_revenue * 0.17)//1000)}K",
        "profitloss": f"${int(profit_loss // 1000)}K",
        "cb": f"{int((total_revenue * 0.17)//1000)}K",
        "runway": f"{int(offset)}K",
        "customers": "99%",
        "arpu": "$600",
        "churn": "13%",
        "monthly_revenue": monthly_revenue,
        "monthly_projected": monthly_projected,
        "quarterly_profit": quarterly_profit,
        "quarterly_loss": quarterly_loss,
    }
