from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup
app = Flask(__name__)
 
# Manual CORS — no flask-cors dependency needed
@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "POST, GET, OPTIONS"
    return response
 
landslide_model = joblib.load("C:/Users/babur/OneDrive/Desktop/ML_floodPrediction/model/landslide_model_v7.pkl")
print("Model loaded OK:", type(landslide_model))
 
FEATURE_COLS = ['rainfall_7day', 'slope', 'elevation', 'soil_moisture', 'ndvi', 'aspect']
 
DISTRICT_ELEVATION = {
    "Shimla":              2206,
    "Manali (Kullu)":      2050,
    "Manali":              2050,
    "Dharamsala (Kangra)": 1457,
    "Dharamsala":          1457,
    "Kangra":              1450,
    "Kullu":               1220,
    "Solan":               1350,
    "Mandi":                850,
}
 
THRESHOLD_HIGH   = 0.65   # only flag HIGH when model is very confident
THRESHOLD_MEDIUM = 0.40
 
RISK_MESSAGES = {
    "LOW":    "Conditions are stable. No immediate landslide threat detected.",
    "MEDIUM": "Elevated risk. Avoid vulnerable slopes and monitor rainfall closely.",
    "HIGH":   "HIGH ALERT: Landslide probability is critical. Evacuate if near steep slopes.",
}
 
 
def classify_risk(prob):
    if prob >= THRESHOLD_HIGH:   return "HIGH"
    if prob >= THRESHOLD_MEDIUM: return "MEDIUM"
    return "LOW"
 
# Add this just before classify_risk() call:
# If rainfall is below 40mm, cap probability at MEDIUM level

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200
 
    try:
        data = request.get_json(force=True, silent=True)
        print("Received:", data)
 
        if data is None:
            return jsonify({"error": "No JSON received"}), 400
 
        district      = str(data.get("district", "Shimla"))
        rainfall_7day = float(data.get("rainfall_7day", 0))
        slope         = float(data.get("slope", 20))
        soil_ui       = float(data.get("soil_moisture", 50))
        soil_moisture = (soil_ui / 100.0) * 25.0
        elevation     = DISTRICT_ELEVATION.get(district, 1800)
        ndvi          = 0.45
 
        features = pd.DataFrame(
        [[rainfall_7day, slope, elevation, soil_moisture, ndvi, 180]],
        columns=FEATURE_COLS
            )
 
        probability = float(landslide_model.predict_proba(features)[0][1])
        prediction  = int(landslide_model.predict(features)[0])
        risk_level  = classify_risk(probability)
 
        if rainfall_7day < 5 and slope < 15 and risk_level != "LOW":
            risk_level  = "LOW"
            probability = min(probability, 0.35)
 
        print("Result: prob=", probability, "risk=", risk_level)
 
        return jsonify({
            "risk": {
                "probability": round(probability, 4),
                "prediction":  prediction,
                "risk_level":  risk_level,
                "message":     RISK_MESSAGES[risk_level],
            },
            "debug": {
                "district":       district,
                "elevation_used": elevation,
                "soil_ui":        soil_ui,
                "soil_model":     round(soil_moisture, 2),
                "rainfall_7day":  rainfall_7day,
                "slope":          slope,
            }
        })
 
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
 
 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})
 


@app.route("/highway-warnings", methods=["GET"])
def highway_warnings():
    warnings = []
    
    try:
        # IMD Shimla subdivision warning page
        res = requests.get(
            "https://mausam.imd.gov.in/imd_latest/contents/subdivisionwise-warning_mc.php?id=25",
            timeout=8
        )
        soup = BeautifulSoup(res.text, 'html.parser')
        text = soup.get_text().lower()

        # Map IMD alert keywords to districts
        district_keywords = {
            "Shimla":              ["shimla"],
            "Manali (Kullu)":      ["kullu", "manali"],
            "Dharamsala (Kangra)": ["kangra", "dharamsala"],
            "Kullu":               ["kullu"],
            "Solan":               ["solan"],
            "Mandi":               ["mandi"],
        }

        alert_level = None
        if "red alert" in text:
            alert_level = "RED"
        elif "orange alert" in text:
            alert_level = "ORANGE"
        elif "yellow alert" in text:
            alert_level = "YELLOW"

        if alert_level:
            for district, keywords in district_keywords.items():
                if any(kw in text for kw in keywords):
                    # Map alert level to highway IDs
                    hw_ids = {
                        "Shimla":              ["NH5-SH", "NH22-SH"],
                        "Manali (Kullu)":      ["NH3-MN", "NH154-MN"],
                        "Dharamsala (Kangra)": ["NH503-DH", "NH88-DH"],
                        "Kullu":               ["NH3-KU", "NH154-KU"],
                        "Solan":               ["NH5-SO", "NH7-SO"],
                        "Mandi":               ["NH154-MA", "NH21-MA"],
                    }
                    severity = "Extreme" if alert_level=="RED" else "High" if alert_level=="ORANGE" else "Moderate"
                    for hw_id in hw_ids.get(district, []):
                        warnings.append({
                            "highway_id": hw_id,
                            "message":    f"IMD {alert_level} ALERT for {district}: {severity} rainfall warning. Landslide risk elevated. Check with local authorities before travel.",
                            "source":     "IMD Shimla",
                            "issued_at":  "Today (IMD)"
                        })

    except Exception as e:
        print("IMD scrape failed:", e)
        # Silently fail — ML predictions still work

    return jsonify(warnings)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
 