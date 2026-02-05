# Environmental Monitor & Predictor

An interactive web application that combines **Google Earth Engine (GEE)** satellite data with **Facebook Prophet** Machine Learning to monitor and forecast environmental health indices like NDVI (Vegetation), NDWI (Moisture), and LST (Surface Temperature).

## 🚀 Features
- **Live Monitoring:** Real-time calculation of environmental indices for any coordinate.
- **Visual Heatmaps:** Dynamic NDVI layers overlaid on an interactive map.
- **Automated Datasets:** Generates 5-year historical monthly datasets via GEE.
- **ML Forecasting:** Uses Time-Series forecasting to predict environmental trends for the next 12 months.

## 🛠️ Tech Stack
- **Frontend:** HTML5, CSS3, JavaScript, Leaflet.js (Mapping), Chart.js (Visualizations).
- **Backend:** FastAPI (Python).
- **Satellite Engine:** Google Earth Engine API.
- **Machine Learning:** Facebook Prophet.

## 📋 Prerequisites
1. A **Google Earth Engine** account and a Project ID.
2. Python 3.8+ installed.

## ⚙️ Setup Instructions

### 1. Clone & Install
```bash
# Install dependencies
pip install -r requirements.txt

# run this command in the terminal
uvicorn main:app --reload

double click the index.html file
```

- ### Please click the *Generate ML Model* button after the dataset is completely loaded