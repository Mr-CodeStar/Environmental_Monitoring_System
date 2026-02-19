# Environmental Monitor & Predictor

An interactive web application that combines **Google Earth Engine (GEE)** satellite data with Machine Learning to monitor and forecast environmental health indices like NDVI (Vegetation), NDBI (Built-up), NDWI (Moisture), and LST (Surface Temperature).


## 🏗️ System Architecture & Flow
1.  **Authentication:** Users log in with their GEE Project ID.
2.  **Live Monitoring:** Real-time calculation of NDVI, NDWI, NDBI, and LST using Sentinel-2 and Landsat-8.
3.  **Data Extraction:** Automated retrieval of 10 years of historical monthly data.
4.  **Ensemble ML Training:** Parallel training of Prophet, Random Forest, XGBoost, and Gradient Boost.
5.  **Forecasting:** Selection of the "Winner" model (lowest RMSE) to generate a 12-month prediction dashboard.


## 🛠️ Tech Stack
- **Frontend:** HTML5, CSS3, JavaScript, Leaflet.js (Mapping), Chart.js (Visualizations).
- **Backend:** FastAPI (Python).
- **Satellite Engine:** Google Earth Engine API.
- **Machine Learning:** Facebook Prophet, Random Forest, XGBoost, and Gradient Boost.

## 📋 Prerequisites
1. A **Google Earth Engine** account and a Project ID.
2. Python 3.9+ installed.

## ⚙️ Setup Instructions

### 1. Clone & Install
```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration
```bash
# Create a .env file in the root directory to manage your local database path:
DATABASE_URL=users.db
```

### 3. Running the System
```bash
# Start the FastAPI server using Uvicorn:
uvicorn main:app --reload
```
The server will start at http://127.0.0.1:8000.

Open **auth.html** in your browser to begin.


### 📈 How to Use
- **Login/Register**: Enter your credentials and Google Earth Engine Project ID.

- **Live Analysis**: Use "Monitor Mode" and click any location on the map to see current environmental indices and heatmaps.

- **Generate Dataset**: Switch to "Train ML" mode. Click a location to extract 10 years of historical data. This saves a file named environment_dataset.csv.

- **Run ML Engine**: Click the "Generate ML Model" button. The system will clean the data, train multiple models, and identify the most accurate one.

- **View Forecast**: A new dashboard will open showing historical trends vs. AI-predicted values for the next Year.

### ✨ Key Highlights of the Project
- **Database**: Uses SQLite for secure and lightweight local user management and project ID storage.
- **Security**: Implements .env files for environment configuration and Pydantic for strict data validation and type safety.
- **Visualization**: Utilizes Leaflet.js for interactive geospatial mapping and Chart.js for high-quality, exportable forecast visualizations.
- **Accuracy**: The ml_service.py ensures high-quality results by handling outlier clipping using the IQR method ($Q3 - Q1$) and normalizing data via MinMax Scaling (0 to 1) before model training.


- ### Please click the *Generate ML Model* button after the dataset is completely loaded