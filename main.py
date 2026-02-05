import os
import csv
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from gee_service import initialize_gee, get_environmental_data, get_historical_dataset
from ml_service import predict_trends 

load_dotenv()

app = FastAPI(title="Environmental Monitoring System API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

PROJECT_ID = os.getenv("GEE_PROJECT_ID")
initialize_gee(PROJECT_ID)

@app.get("/")
async def home():
    return {
        "message": "Environmental Monitoring System API is running",
        "endpoints": {
            "monitor": "/api/monitor?lat={lat}&lon={lon}",
            "dataset": "/api/dataset?lat={lat}&lon={lon}&years=5",
            "predict": "/api/predict?index=NDVI",
            "docs": "/docs"
        }
    }

@app.get("/api/monitor")
async def monitor_location(lat: float, lon: float, start: str = "2024-01-01", end: str = "2024-12-31"):
    try:
        data = get_environmental_data(lat, lon, start, end)
        return {"status": "success", "data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset")
async def download_dataset(lat: float, lon: float, years: int = 5):
    try:
        data = get_historical_dataset(lat, lon, years)
        
        if not data:
            return {"status": "error", "message": "No data found for this location"}

        csv_file = "environment_dataset.csv"
        keys = data[0].keys()
        with open(csv_file, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

        return {
            "status": "success",
            "message": f"Dataset saved to {csv_file}",
            "total_records": len(data),
            "records": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/predict")
async def get_prediction(index: str = "NDVI"):
    try:
        # Calls the function from ml_service.py
        result = predict_trends(index)
        if "error" in result:
            # Handle cases where CSV doesn't exist or data is too sparse
            raise HTTPException(status_code=400, detail=result["error"])
            
        return {
            "status": "success",
            "index": index,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
