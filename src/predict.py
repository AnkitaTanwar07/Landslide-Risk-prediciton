import joblib
import pandas as pd
import os
import sys

# =========================
# FIX TERMINAL ENCODING (optional safety)
# =========================
sys.stdout.reconfigure(encoding='utf-8')

# =========================
# 1. LOAD MODEL (ROBUST PATH)
# =========================
base_path = os.path.dirname(os.path.dirname(__file__))
model_path = os.path.join(base_path, "model", "landslide_model.pkl")

model = joblib.load(model_path)

# =========================
# 2. PREDICTION FUNCTION
# =========================
def predict_landslide(lat, lon, rainfall, slope, elevation):
    
    # Create DataFrame with proper feature names (no warnings)
    features = pd.DataFrame([{
        "latitude": lat,
        "longitude": lon,
        "rainfall": rainfall,
        "slope": slope,
        "elevation": elevation
    }])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # probability of landslide
    
    # =========================
    # 3. RISK LOGIC
    # =========================
    if probability < 0.3:
        risk = "LOW"
        message = "Safe conditions. No immediate landslide risk."
    
    elif probability < 0.7:
        risk = "MEDIUM"
        message = "Caution advised. Conditions may become unstable."
    
    else:
        risk = "HIGH"
        message = "High landslide risk! Avoid travel in this area."
    
    return {
        "prediction": int(prediction),
        "probability": round(probability, 2),
        "risk_level": risk,
        "message": message
    }

# =========================
# 4. TEST RUN
# =========================
if __name__ == "__main__":
    
    result = predict_landslide(
        lat=31.5,
        lon=77.2,
        rainfall=5.0,
        slope=25,
        elevation=2000
    )
    
    print("\n===== LANDSLIDE RISK RESULT =====")
    print(f"Prediction     : {result['prediction']}")
    print(f"Probability    : {result['probability']}")
    print(f"Risk Level     : {result['risk_level']}")
    print(f"Message        : {result['message']}")