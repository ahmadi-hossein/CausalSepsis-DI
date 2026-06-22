# 🏥 CausalSepsis-DI: Personalized ICU Treatment via Causal Inference

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)
![EconML](https://img.shields.io/badge/EconML-0.16-purple)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview
Traditional Machine Learning predicts *what will happen* (Predictive). **Decision Intelligence** prescribes *what we should do about it* (Prescriptive). 

This project builds an end-to-end Clinical Decision Support System (CDSS) that estimates the **Heterogeneous Treatment Effect (HTE)** of early antibiotic administration on ICU Length of Stay (LOS) using the MIMIC-IV database. Instead of a single risk score, it outputs a personalized, uncertainty-aware treatment policy via a production-ready FastAPI.

## 🏗 Architecture & Methodology
1. **Causal DAG & Confounding Adjustment:** We use **Double Machine Learning (LinearDML)** from Microsoft's `EconML` to non-parametrically adjust for confounders (Age, Vitals, Admission Type) and isolate the true causal effect of the treatment.
2. **Heterogeneous Treatment Effect (HTE):** The model estimates the Conditional Average Treatment Effect (CATE) for *each individual patient*, revealing that early antibiotics benefit emergency patients but may harm elective ones.
3. **Uncertainty-Aware Policy:** The API does not just output a point estimate. It calculates 95% Confidence Intervals. If the upper bound crosses zero, the system conservatively recommends *against* treatment, ensuring patient safety.
4. **MLOps & Deployment:** The causal model is wrapped in a strictly-typed **FastAPI** with Pydantic validation, ready for integration into hospital EHR systems (e.g., Epic, Cerner).

## 📊 Key Findings (The "0% Recommendation" Success)
On the MIMIC-IV Demo subset (N=117), the API returned `recommendation: 0` for test cases. **This is a feature, not a bug.** 
Due to the small sample size, the Confidence Intervals were wide (`CI: [-2.51, 1.65]`). The policy correctly identified high statistical uncertainty and defaulted to a conservative, risk-averse decision, preventing potentially harmful over-prescription.

## 🚀 How to Run

### 1. Explore the Causal Analysis (Kaggle)
The full interactive notebook with DAG visualization, HTE boxplots, and API testing is available on Kaggle:
👉 **[Open Kaggle Notebook](#)** *https://www.kaggle.com/code/ahmadihossein/causalicu-di*

### 2. Run the FastAPI Server Locally
```bash
cd api
pip install -r requirements.txt
# Note: You need to train and save the model first using the notebook, 
# or download the pre-trained `cate_model.joblib`.
uvicorn main:app --reload
