
#### ۲. فایل `api/main.py` (کد پروداکشن FastAPI)
```python
# api/main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(
    title="CausalICU-DI API",
    description="Uncertainty-aware treatment recommendation engine.",
    version="1.0.0"
)

# Load the pre-trained causal model (Assuming you saved it via joblib.dump(cate_model, 'cate_model.joblib'))
# cate_model = joblib.load("cate_model.joblib") 

class PatientInput(BaseModel):
    anchor_age: int = Field(..., ge=18, le=120)
    gender: int = Field(..., ge=0, le=1)
    is_emergency: int = Field(..., ge=0, le=1)
    hr_first12h: float = Field(..., ge=30, le=200)
    map_first12h: float = Field(..., ge=30, le=180)

class CausalOutput(BaseModel):
    cate_days: float
    ci_lower: float
    ci_upper: float
    recommendation: int
    confidence: str

@app.post("/predict", response_model=CausalOutput)
def predict(patient: PatientInput):
    X_in = np.array([[patient.anchor_age, patient.gender, patient.is_emergency, 
                      patient.hr_first12h, patient.map_first12h]])
    
    # In production, uncomment the lines below:
    # cate = float(cate_model.effect(X_in).item())
    # lower, upper = cate_model.effect_interval(X_in, alpha=0.05)
    # l_val, u_val = float(lower.item()), float(upper.item())
    
    # Mock response for demonstration without the model file:
    cate, l_val, u_val = -0.43, -2.518, 1.659 
    
    rec = 1 if (cate < -0.5 and u_val < 0) else 0
    width = u_val - l_val
    conf = "High" if width < 1.0 else ("Medium" if width < 2.5 else "Low")
    
    return CausalOutput(cate_days=round(cate, 3), ci_lower=round(l_val, 3), 
                        ci_upper=round(u_val, 3), recommendation=rec, confidence=conf)