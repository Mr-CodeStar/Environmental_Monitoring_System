import os
import csv
import sqlite3
import webbrowser
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta

from gee_service import initialize_gee, get_environmental_data, get_historical_dataset
from ml_service import predict_all_trends

# Load environment variables
load_dotenv()

app = FastAPI(title="Environmental Monitoring System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATABASE SETUP ---
DB_NAME = os.getenv("DATABASE_URL", "users.db")

def get_db_connection():
    return sqlite3.connect(DB_NAME)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            gee_project_id TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# --- MODELS ---
class UserAuth(BaseModel):
    username: str
    password: str
    gee_project_id: Optional[str] = None

# --- AUTH ENDPOINTS ---
@app.post("/register")
async def register(user: UserAuth):
    if not user.gee_project_id:
        raise HTTPException(status_code=400, detail="GEE Project ID is required for registration")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password, gee_project_id) VALUES (?, ?, ?)", 
                       (user.username, user.password, user.gee_project_id))
        conn.commit()
        conn.close()
        return {"message": "User registered successfully"}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Username already exists")
    except Exception:
        raise HTTPException(status_code=500, detail="Invalid Username or Password")

@app.post("/login")
async def login(user: UserAuth):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT gee_project_id FROM users WHERE username = ? AND password = ?", 
                   (user.username, user.password))
    result = cursor.fetchone()
    conn.close()

    if result:
        user_gee_id = result[0]
        initialize_gee(user_gee_id)
        
        file_path = os.path.abspath("index.html")
        web_url = f"file://{file_path}"
        webbrowser.open(web_url, new=0)
        
        return {"status": "success", "message": "Login successful"}
    else:
        raise HTTPException(status_code=401, detail="Invalid Username or Password")

@app.get("/")
async def home():
    return {"message": "Environmental Monitoring System API is running"}

@app.get("/monitor")
async def monitor_location(
    lat: float, 
    lon: float, 
    start: Optional[str] = None, 
    end: Optional[str] = None
):
    try:
        if not end:
            end = datetime.now().strftime('%Y-%m-%d')
        if not start:
            start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
        data = get_environmental_data(lat, lon, start, end)
        return {
            "status": "success", 
            "data": data,
            "date_range": {"start": start, "end": end}
        }
    except Exception as e:
        print(f"Error in /monitor: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset")
async def download_dataset(lat: float, lon: float, years: int = 10):
    try:
        data = get_historical_dataset(lat, lon, years)
        if not data:
            return {"status": "error", "message": "No data found"}

        csv_file = "environment_dataset.csv"
        keys = data[0].keys()
        with open(csv_file, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

        return {"status": "success", "message": f"Dataset saved to {csv_file}", "total_records": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict")
async def get_all_predictions():
    try:
        results = predict_all_trends()
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
        return {"status": "success", "data": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))