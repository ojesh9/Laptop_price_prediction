from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load your trained pipeline model (make sure this file exists in project root)
MODEL_PATH = "best_laptop_price_model.joblib"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Put your trained model at {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

# Features expected by the model (from your pipeline)
FEATURE_ORDER = [
    "Inches",
    "CPU_Frequency",
    "RAM (GB)",
    "SSD_GB",
    "HDD_GB",
    "Total_Storage_GB",
    "Weight (kg)",
    "Company",
    "TypeName",
    "CPU_Company",
    "CPU_Type",
    "GPU_Company",
    "GPU_Type",
    "OpSys",
    "ScreenResolution"
]


# Helper: parse numbers safely
def to_float(x, default=np.nan):
    try:
        if x is None or x=="":
            return default
        return float(str(x).strip())
    except:
        return default

@app.route("/")
def index():
    # For nice UI we include a few default dropdown options.
    # You can replace these lists by extracting unique values from your CSV (instructions below).
    companies = ["Dell","HP","Lenovo","Acer","Asus","Apple","MSI","Microsoft","Other"]
    types = ["Ultrabook","Notebook","Gaming","2-in-1 Convertible","Workstation","Netbook","Other"]
    cpu_companies = ["Intel","AMD","Apple","Other"]
    gpu_companies = ["NVIDIA","Intel","AMD","Other"]
    ops = ["Windows","Mac","Linux","Chrome OS","Other"]
    
    return render_template("index.html",
                           companies=companies,
                           types=types,
                           cpu_companies=cpu_companies,
                           gpu_companies=gpu_companies,
                           ops=ops)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects form-data from the UI:
      numeric fields: Inches, CPU_Frequency, RAM, SSD_GB, HDD_GB, Weight
      categorical: Company, TypeName, CPU_Company, CPU_Type, GPU_Company, GPU_Type, OpSys
    """
    data = request.json if request.is_json else request.form
    # Read numeric inputs
    inches = to_float(data.get("Inches"))
    cpu_freq = to_float(data.get("CPU_Frequency"))
    ram = to_float(data.get("RAM_GB") or data.get("RAM (GB)") or data.get("RAM"))
    ssd_gb = to_float(data.get("SSD_GB")) or 0.0
    hdd_gb = to_float(data.get("HDD_GB")) or 0.0
    weight = to_float(data.get("Weight_kg") or data.get("Weight (kg)") or data.get("Weight"))

    total_storage = (ssd_gb if not np.isnan(ssd_gb) else 0.0) + (hdd_gb if not np.isnan(hdd_gb) else 0.0)

    # Categorical (fall back to "Other" if empty)
    company = data.get("Company") or "Other"
    typename = data.get("TypeName") or "Other"
    cpu_company = data.get("CPU_Company") or "Other"
    cpu_type = data.get("CPU_Type") or "Other"
    gpu_company = data.get("GPU_Company") or "Other"
    gpu_type = data.get("GPU_Type") or "Other"
    opsys = data.get("OpSys") or "Other"

    # Build dataframe with the expected columns and order:
    record = {
        "Inches": inches,
        "CPU_Frequency": cpu_freq,
        "RAM (GB)": ram,
        "SSD_GB": ssd_gb,
        "HDD_GB": hdd_gb,
        "Total_Storage_GB": total_storage,
        "Weight (kg)": weight,
        "Company": company,
        "TypeName": typename,
        "CPU_Company": cpu_company,
        "CPU_Type": cpu_type,
        "GPU_Company": gpu_company,
        "GPU_Type": gpu_type,
        "OpSys": opsys ,
        "ScreenResolution": data.get("ScreenResolution") or "1920x1080"
    }

    X = pd.DataFrame([record], columns=FEATURE_ORDER)

    try:
        pred = model.predict(X)[0]
        price_eur = float(pred)

        eur_to_inr = 90
        price_inr = price_eur 

        return jsonify({
            "success": True,
            "predicted_price": round(price_inr, 2),  # <-- FIXED
            "currency": "INR"
    })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
