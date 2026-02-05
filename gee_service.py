import ee
import pandas as pd

def initialize_gee(project_id: str):
    """Initializes GEE with a specific project ID."""
    try:
        ee.Initialize(project=project_id)
        print("Initialization successful")
        return True
    except Exception as e:
        print(f"Failed to initialize GEE: {e}")
        return False

def get_environmental_data(lat: float, lon: float, start_date: str, end_date: str):
    """
    Returns both numerical indices and Leaflet-compatible tile URLs.
    """
    poi = ee.Geometry.Point([lon, lat])
    roi = poi.buffer(5000).bounds()

    # --- 1. Load Data ---
    s2_col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(roi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)))
    
    l8_col = (ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
              .filterBounds(roi)
              .filterDate(start_date, end_date))

    # Check if images exist for the single monitor request
    if s2_col.size().getInfo() == 0 or l8_col.size().getInfo() == 0:
        raise Exception("No satellite data available for this date range/location.")

    s2_img = s2_col.median().clip(roi)
    l8_img = l8_col.median().clip(roi)

    # --- 2. Calculate Numerical Indices ---
    ndvi = s2_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = s2_img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndbi = s2_img.normalizedDifference(['B11', 'B8']).rename('NDBI')
    lst = l8_img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')

    # Extract point values
    combined_stats = ndvi.addBands([ndwi, ndbi, lst])
    stats = combined_stats.sample(poi, 30).first().getInfo()

    # --- 3. Generate Visual Map Tiles ---
    rgb_vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
    map_id_rgb = s2_img.getMapId(rgb_vis)

    ndvi_vis = {'min': 0, 'max': 0.8, 'palette': ['red', 'yellow', 'green']}
    map_id_ndvi = ndvi.getMapId(ndvi_vis)

    return {
        "indices": stats['properties'],
        "tiles": {
            "true_color": map_id_rgb['tile_fetcher'].url_format,
            "ndvi_heatmap": map_id_ndvi['tile_fetcher'].url_format
        }
    }

def get_historical_dataset(lat, lon, years=5):
    """
    Fetches monthly environmental data for the last 'X' years.
    Includes safety checks to prevent 'No bands' errors.
    """
    poi = ee.Geometry.Point([lon, lat])
    end_date = ee.Date(pd.Timestamp.now().strftime('%Y-%m-%d'))
    start_date = end_date.advance(-years, 'year')
    
    n_months = end_date.difference(start_date, 'month').round().getInfo()
    
    dataset = []
    print(f"Generating dataset for {n_months} months...")

    for m in range(n_months):
        try:
            m_start = start_date.advance(m, 'month')
            m_end = m_start.advance(1, 'month')
            
            # Filter collections
            s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(poi).filterDate(m_start, m_end)
            l8_col = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterBounds(poi).filterDate(m_start, m_end)
            
            # Check if images actually exist for this specific month
            if s2_col.size().getInfo() > 0 and l8_col.size().getInfo() > 0:
                s2_img = s2_col.median()
                l8_img = l8_col.median()
                
                # Band Math
                ndvi = s2_img.normalizedDifference(['B8', 'B4']).rename('NDVI')
                ndwi = s2_img.normalizedDifference(['B3', 'B8']).rename('NDWI')
                
                # Surface Temperature logic
                lst = l8_img.select('ST_B10').multiply(0.00341802).add(149.0).subtract(273.15).rename('LST')
                
                combined = ndvi.addBands([ndwi, lst])
                val = combined.sample(poi, 30).first().getInfo()
                
                if val and 'properties' in val:
                    data_row = val['properties']
                    data_row['date'] = m_start.format('YYYY-MM-dd').getInfo()
                    dataset.append(data_row)
            else:
                # Skip month if no data
                continue
                
        except Exception as e:
            print(f"Skipping month {m} due to error: {e}")
            continue

    return dataset