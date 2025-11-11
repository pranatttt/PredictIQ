import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

DB_URI = os.getenv("DB_URI")
if not DB_URI:
    raise ValueError("❌ DB_URI not found. Add it to your .env file.")

engine = create_engine(DB_URI)

# Load CSV safely (relative path recommended)
df = pd.read_csv("Backend/Data/pred_sales_data_2.csv")

df.to_sql(
    'pred_sales_data_2',
    con=engine,
    if_exists='append',
    index=False
)

print("✅ Data uploaded to pred_sales_data_2")
