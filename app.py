import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── 1. FastAPI Application Instance (Vercel Entrypoint) ─────────────────
app = FastAPI(
    title="Telco Customer Churn Predictor",
    description="Machine Learning inference service for customer churn prediction",
    version="1.0.0"
)

# ── 2. Load Model & Threshold ────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_pipeline.joblib")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "churn_threshold.joblib")

pipeline = None
threshold = 0.3114

if os.path.exists(MODEL_PATH) and os.path.exists(THRESHOLD_PATH):
    try:
        pipeline = joblib.load(MODEL_PATH)
        threshold = float(joblib.load(THRESHOLD_PATH))
    except Exception as e:
        print(f"Error loading model artifacts: {e}")

# ── 3. Pydantic Customer Input Schema ────────────────────────────────────
class CustomerData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "Yes"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 840.0

def compute_churn_risk(data: CustomerData):
    if pipeline is None:
        raise ValueError("Model pipeline not loaded.")

    input_df = pd.DataFrame([{
        "gender": data.gender,
        "SeniorCitizen": int(data.SeniorCitizen),
        "Partner": data.Partner,
        "Dependents": data.Dependents,
        "tenure": int(data.tenure),
        "PhoneService": data.PhoneService,
        "MultipleLines": data.MultipleLines,
        "InternetService": data.InternetService,
        "OnlineSecurity": data.OnlineSecurity,
        "OnlineBackup": data.OnlineBackup,
        "DeviceProtection": data.DeviceProtection,
        "TechSupport": data.TechSupport,
        "StreamingTV": data.StreamingTV,
        "StreamingMovies": data.StreamingMovies,
        "Contract": data.Contract,
        "PaperlessBilling": data.PaperlessBilling,
        "PaymentMethod": data.PaymentMethod,
        "MonthlyCharges": float(data.MonthlyCharges),
        "TotalCharges": float(data.TotalCharges),
    }])

    prob = float(pipeline.predict_proba(input_df)[0, 1])
    is_high_risk = prob >= threshold

    # Factor Attribution
    risk_drivers = []
    protective_factors = []

    if data.Contract == "Month-to-month":
        risk_drivers.append("Month-to-month Contract: High churn vulnerability (~43%) without long-term lock-in.")
    else:
        protective_factors.append(f"{data.Contract} Contract: Strong contractual commitment significantly protects retention.")

    if data.tenure <= 6:
        risk_drivers.append(f"Early Tenure ({data.tenure} mos): Peak customer hazard window (months 1–6).")
    elif data.tenure >= 24:
        protective_factors.append(f"Mature Tenure ({data.tenure} mos): Established account with high survival probability.")

    if data.InternetService == "Fiber optic":
        risk_drivers.append("Fiber Optic Internet: High churn service cohort (~42%) sensitive to cost & downtime.")
    elif data.InternetService == "DSL":
        protective_factors.append("DSL Internet: Lower churn rate (~19%) and stable price point.")

    if data.PaymentMethod == "Electronic check":
        risk_drivers.append("Electronic Check: High-friction manual payment method with 45.3% churn rate.")
    elif "automatic" in data.PaymentMethod:
        protective_factors.append(f"Auto-Pay ({data.PaymentMethod.split(' ')[0]}): Automated billing reduces monthly payment friction.")

    if data.MonthlyCharges >= 75.0:
        risk_drivers.append(f"High Monthly Charges (${data.MonthlyCharges:.2f}): Elevated billing heightens price sensitivity.")
    elif data.MonthlyCharges <= 45.0:
        protective_factors.append(f"Low Monthly Charges (${data.MonthlyCharges:.2f}): Budget-friendly plan minimizes bill shock.")

    if data.InternetService != "No" and data.TechSupport == "No":
        risk_drivers.append("No Tech Support: Missing technical assistance increases defection during service issues.")
    if data.InternetService != "No" and data.OnlineSecurity == "No":
        risk_drivers.append("No Online Security: Lacks bundled security protection.")

    # Recommendation
    if is_high_risk:
        if data.Contract == "Month-to-month":
            recommendation = "Offer a 10%–15% annual contract discount or free streaming perk to secure a 1-year agreement."
        elif data.PaymentMethod == "Electronic check":
            recommendation = "Provide a one-time $10 bill credit to enroll in automated Credit Card or Bank Transfer billing."
        else:
            recommendation = "Schedule an account review with senior customer success to address service quality."
    else:
        recommendation = "Customer is stable. Maintain standard loyalty engagement and monitor regular service usage."

    return {
        "churn_probability": round(prob, 4),
        "threshold": round(threshold, 4),
        "is_high_risk": is_high_risk,
        "risk_label": "HIGH" if is_high_risk else "LOW",
        "risk_drivers": risk_drivers,
        "protective_factors": protective_factors,
        "recommended_action": recommendation
    }

# ── 4. API Endpoints ─────────────────────────────────────────────────────
@app.post("/predict")
@app.post("/api/predict")
async def predict_endpoint(customer: CustomerData):
    try:
        res = compute_churn_risk(customer)
        return JSONResponse(content=res)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "threshold": threshold
    }

# ── 5. Interactive Frontend (Served at GET /) ────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Telco Customer Churn Risk Predictor</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body {{ font-family: 'Plus Jakarta Sans', sans-serif; }}
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen py-8 px-4 sm:px-6 lg:px-8">
  <div class="max-w-6xl mx-auto space-y-8">
    
    <!-- Header -->
    <div class="bg-slate-800/80 backdrop-blur border border-slate-700/60 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
      <div>
        <div class="flex items-center gap-3">
          <span class="text-3xl">📡</span>
          <h1 class="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">Telco Customer Churn Predictor</h1>
        </div>
        <p class="text-slate-400 mt-1 text-sm sm:text-base">Real-time ML risk scoring & factor attribution powered by Tuned XGBoost</p>
      </div>
      <div class="flex items-center gap-2 bg-slate-900/60 border border-slate-700 px-3 py-1.5 rounded-xl text-xs font-medium text-slate-300">
        <span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
        Model Calibrated Cutoff: <strong class="text-white">{(threshold * 100):.1f}%</strong>
      </div>
    </div>

    <!-- Main Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
      
      <!-- Input Panel (2 cols) -->
      <div class="lg:col-span-2 bg-slate-800/60 border border-slate-700/50 rounded-2xl p-6 shadow-xl space-y-6">
        <h2 class="text-lg font-bold text-white flex items-center gap-2 border-b border-slate-700 pb-3">
          <span>📋</span> Customer Attributes & Service Configuration
        </h2>

        <form id="churnForm" class="space-y-6">
          
          <!-- Category 1: Contract & Billing -->
          <div>
            <h3 class="text-xs font-semibold uppercase tracking-wider text-cyan-400 mb-3">1. Contract & Billing Details</h3>
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Contract Type</label>
                <select id="Contract" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="Month-to-month" selected>Month-to-month</option>
                  <option value="One year">One year</option>
                  <option value="Two year">Two year</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Payment Method</label>
                <select id="PaymentMethod" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="Electronic check" selected>Electronic check</option>
                  <option value="Mailed check">Mailed check</option>
                  <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
                  <option value="Credit card (automatic)">Credit card (automatic)</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Tenure: <span id="tenureVal" class="text-cyan-400 font-bold">12</span> months</label>
                <input id="tenure" type="range" min="0" max="72" value="12" class="w-full accent-cyan-500 cursor-pointer">
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Monthly Charges: $<span id="monthlyVal" class="text-cyan-400 font-bold">70.00</span></label>
                <input id="MonthlyCharges" type="range" min="18" max="120" step="0.5" value="70.0" class="w-full accent-cyan-500 cursor-pointer">
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Total Charges ($)</label>
                <input id="TotalCharges" type="number" step="10" value="840.0" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Paperless Billing</label>
                <select id="PaperlessBilling" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="Yes" selected>Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Category 2: Core Telecom Services -->
          <div class="border-t border-slate-700/60 pt-4">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-cyan-400 mb-3">2. Core & Security Services</h3>
            <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Internet Service</label>
                <select id="InternetService" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="Fiber optic" selected>Fiber optic</option>
                  <option value="DSL">DSL</option>
                  <option value="No">No</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Online Security</label>
                <select id="OnlineSecurity" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="No" selected>No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Tech Support</label>
                <select id="TechSupport" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="No" selected>No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Online Backup</label>
                <select id="OnlineBackup" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="No" selected>No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Device Protection</label>
                <select id="DeviceProtection" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="No" selected>No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Phone Service</label>
                <select id="PhoneService" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-cyan-500">
                  <option value="Yes" selected>Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
            </div>
          </div>

          <!-- Category 3: Entertainment & Demographics -->
          <div class="border-t border-slate-700/60 pt-4">
            <h3 class="text-xs font-semibold uppercase tracking-wider text-cyan-400 mb-3">3. Entertainment & Demographics</h3>
            <div class="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Streaming TV</label>
                <select id="StreamingTV" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs sm:text-sm text-white">
                  <option value="No" selected>No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Streaming Movies</label>
                <select id="StreamingMovies" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs sm:text-sm text-white">
                  <option value="No" selected>No</option>
                  <option value="Yes">Yes</option>
                  <option value="No internet service">No internet service</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Senior Citizen</label>
                <select id="SeniorCitizen" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs sm:text-sm text-white">
                  <option value="0" selected>No (0)</option>
                  <option value="1">Yes (1)</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Partner</label>
                <select id="Partner" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs sm:text-sm text-white">
                  <option value="Yes" selected>Yes</option>
                  <option value="No">No</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Dependents</label>
                <select id="Dependents" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs sm:text-sm text-white">
                  <option value="No" selected>No</option>
                  <option value="Yes">Yes</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Gender</label>
                <select id="gender" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs sm:text-sm text-white">
                  <option value="Female" selected>Female</option>
                  <option value="Male">Male</option>
                </select>
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Multiple Lines</label>
                <select id="MultipleLines" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-2 text-xs sm:text-sm text-white">
                  <option value="No" selected>No</option>
                  <option value="Yes">Yes</option>
                  <option value="No phone service">No phone service</option>
                </select>
              </div>
            </div>
          </div>

          <button type="button" id="predictBtn" class="w-full py-3.5 bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-white font-bold rounded-xl shadow-lg transition duration-200 flex items-center justify-center gap-2">
            <span>🚀</span> Analyze Customer Churn Risk
          </button>
        </form>
      </div>

      <!-- Result Card (1 col) -->
      <div class="bg-slate-800/60 border border-slate-700/50 rounded-2xl p-6 shadow-xl flex flex-col justify-between space-y-6">
        <div>
          <h2 class="text-lg font-bold text-white border-b border-slate-700 pb-3 flex items-center gap-2">
            <span>🎯</span> Prediction & Attribution
          </h2>

          <div id="resultsPlaceholder" class="py-12 text-center text-slate-400 space-y-3">
            <div class="text-4xl">📊</div>
            <p class="text-sm">Adjust customer attributes and click <strong class="text-slate-200">Analyze Churn Risk</strong> to score this account.</p>
          </div>

          <div id="resultsContent" class="hidden space-y-6 mt-4">
            
            <!-- Risk Badge & Score -->
            <div id="riskCard" class="p-5 rounded-2xl border text-center transition-all">
              <span id="riskBadge" class="inline-block px-3 py-1 text-xs font-extrabold uppercase rounded-full mb-2"></span>
              <div class="text-4xl font-extrabold text-white" id="probText">0.0%</div>
              <p class="text-xs text-slate-400 mt-1">Predicted Churn Probability</p>

              <!-- Progress bar -->
              <div class="w-full bg-slate-900 rounded-full h-2.5 mt-4 overflow-hidden border border-slate-700/60">
                <div id="probBar" class="h-2.5 rounded-full transition-all duration-500" style="width: 0%"></div>
              </div>
            </div>

            <!-- Risk Drivers -->
            <div class="space-y-3 text-xs">
              <h4 class="font-bold uppercase tracking-wider text-rose-400 flex items-center gap-1.5">
                <span>⚠️</span> Risk-Elevating Factors
              </h4>
              <ul id="riskDriversList" class="space-y-1.5 text-slate-300"></ul>

              <h4 class="font-bold uppercase tracking-wider text-emerald-400 pt-2 flex items-center gap-1.5">
                <span>🛡️</span> Retention-Protecting Buffers
              </h4>
              <ul id="protectiveFactorsList" class="space-y-1.5 text-slate-300"></ul>
            </div>

            <!-- Recommendation Card -->
            <div class="p-4 bg-slate-900/80 border border-slate-700/80 rounded-xl space-y-1">
              <div class="text-xs font-bold text-amber-400 flex items-center gap-1">
                <span>💡</span> Recommended Action
              </div>
              <p id="recommendationText" class="text-xs text-slate-200 leading-relaxed"></p>
            </div>

          </div>
        </div>

        <div class="pt-4 border-t border-slate-700/50 text-[11px] text-slate-400 flex items-center justify-between">
          <span>Engine: Tuned XGBoost</span>
          <span>Cutoff: {(threshold * 100):.1f}%</span>
        </div>
      </div>

    </div>

  </div>

  <script>
    const tenureSlider = document.getElementById('tenure');
    const tenureVal = document.getElementById('tenureVal');
    const monthlySlider = document.getElementById('MonthlyCharges');
    const monthlyVal = document.getElementById('monthlyVal');
    const totalInput = document.getElementById('TotalCharges');

    tenureSlider.addEventListener('input', (e) => {{
      tenureVal.textContent = e.target.value;
      totalInput.value = (parseFloat(e.target.value) * parseFloat(monthlySlider.value)).toFixed(2);
    }});

    monthlySlider.addEventListener('input', (e) => {{
      monthlyVal.textContent = parseFloat(e.target.value).toFixed(2);
      totalInput.value = (parseFloat(tenureSlider.value) * parseFloat(e.target.value)).toFixed(2);
    }});

    document.getElementById('predictBtn').addEventListener('click', async () => {{
      const payload = {{
        gender: document.getElementById('gender').value,
        SeniorCitizen: parseInt(document.getElementById('SeniorCitizen').value),
        Partner: document.getElementById('Partner').value,
        Dependents: document.getElementById('Dependents').value,
        tenure: parseInt(document.getElementById('tenure').value),
        PhoneService: document.getElementById('PhoneService').value,
        MultipleLines: document.getElementById('MultipleLines').value,
        InternetService: document.getElementById('InternetService').value,
        OnlineSecurity: document.getElementById('OnlineSecurity').value,
        OnlineBackup: document.getElementById('OnlineBackup').value,
        DeviceProtection: document.getElementById('DeviceProtection').value,
        TechSupport: document.getElementById('TechSupport').value,
        StreamingTV: document.getElementById('StreamingTV').value,
        StreamingMovies: document.getElementById('StreamingMovies').value,
        Contract: document.getElementById('Contract').value,
        PaperlessBilling: document.getElementById('PaperlessBilling').value,
        PaymentMethod: document.getElementById('PaymentMethod').value,
        MonthlyCharges: parseFloat(document.getElementById('MonthlyCharges').value),
        TotalCharges: parseFloat(document.getElementById('TotalCharges').value)
      }};

      try {{
        const res = await fetch('/predict', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(payload)
        }});
        const data = await res.json();

        document.getElementById('resultsPlaceholder').classList.add('hidden');
        document.getElementById('resultsContent').classList.remove('hidden');

        const prob = data.churn_probability;
        const pct = (prob * 100).toFixed(1) + '%';
        document.getElementById('probText').textContent = pct;

        const bar = document.getElementById('probBar');
        bar.style.width = Math.min(prob * 100, 100) + '%';

        const card = document.getElementById('riskCard');
        const badge = document.getElementById('riskBadge');

        if (data.is_high_risk) {{
          card.className = 'p-5 rounded-2xl border bg-rose-950/30 border-rose-600/50 text-center';
          badge.className = 'inline-block px-3 py-1 text-xs font-extrabold uppercase rounded-full mb-2 bg-rose-500/20 text-rose-300 border border-rose-500/40';
          badge.textContent = '🔴 High Churn Risk';
          bar.className = 'h-2.5 rounded-full bg-rose-500 transition-all duration-500';
        }} else {{
          card.className = 'p-5 rounded-2xl border bg-emerald-950/30 border-emerald-600/50 text-center';
          badge.className = 'inline-block px-3 py-1 text-xs font-extrabold uppercase rounded-full mb-2 bg-emerald-500/20 text-emerald-300 border border-emerald-500/40';
          badge.textContent = '🟢 Low Churn Risk';
          bar.className = 'h-2.5 rounded-full bg-emerald-500 transition-all duration-500';
        }}

        // Drivers
        const rList = document.getElementById('riskDriversList');
        rList.innerHTML = '';
        if (data.risk_drivers.length > 0) {{
          data.risk_drivers.forEach(d => {{
            const li = document.createElement('li');
            li.textContent = '• ' + d;
            rList.appendChild(li);
          }});
        }} else {{
          rList.innerHTML = '<li class=\"text-slate-400 italic\">No major risk indicators detected.</li>';
        }}

        // Protective
        const pList = document.getElementById('protectiveFactorsList');
        pList.innerHTML = '';
        if (data.protective_factors.length > 0) {{
          data.protective_factors.forEach(p => {{
            const li = document.createElement('li');
            li.textContent = '• ' + p;
            pList.appendChild(li);
          }});
        }} else {{
          pList.innerHTML = '<li class=\"text-slate-400 italic\">No significant protective buffers.</li>';
        }}

        // Recommendation
        document.getElementById('recommendationText').textContent = data.recommended_action;

      }} catch (err) {{
        alert('Prediction request failed: ' + err);
      }}
    }});
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
