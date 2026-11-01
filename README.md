# 🛡️ Predictive Forecasting of Care Load & Placement Demand

A predictive intelligence and decision-support web application developed for the HHS Unaccompanied Alien Children (UAC) program. This tool enables government stakeholders and program managers to forecast daily care volumes, evaluate operational throughput pressures, monitor model error metrics, and run scenario stress-tests against border transfer surges and placement bottlenecks.

---

## 🚀 Live Demo
* **Streamlit Web Application**: [Access the Live Dashboard Here](https://predictive-forecasting-of-care-load-placement-demand-atexlrcbx.streamlit.app/)

---

## 📌 Project Overview
Managing shelter capacity and sponsor placement logistics requires timely foresight. This project builds a complete end-to-end data pipeline and interactive forecasting dashboard to:
1. **Forecast Future Care Loads**: Predict short-term care trajectories using baseline statistical models and advanced machine learning models (Random Forest).
2. **Monitor Discharge Demand (Net Pressure)**: Track daily imbalances between incoming children transferred from CBP custody and successful sponsor discharges.
3. **Simulate Scenario Stress-Shocks**: Stress-test shelter capacity against percentage surges in CBP transfers and administrative placement/discharge bottlenecks.
4. **Evaluate Model Performance**: Compare error rates (MAE, RMSE, MAPE) across multiple forecasting approaches.

---

## 📂 Repository Structure
```text
├── HHS_Unaccompanied_Alien_Children_Program.csv   # Historical program dataset
├── streamlit_app.py                             # Main Streamlit web application script
├── requirements.txt                             # Python package dependencies
└── README.md                                    # Project documentation

---

📊 Key Dashboard Modules
Tab 1: Care Load Forecast & Scenarios
Displays historical trends alongside future predictions (with 95% confidence interval uncertainty bounds).
Integrates interactive sidebar controls to adjust forecast horizons (7 to 30 days) and simulate custom operational stress scenarios.

Tab 2: Discharge Demand Panel
Features a net daily care pressure bar chart highlighting intake vs. exit imbalances.

Tab 3: Model Performance Evaluation
Summarizes validation metrics comparing Naïve Persistence, 7-Day Moving Average, and Random Forest Regressor models.

---

📦 Requirements (requirements.txt)
streamlit
pandas
numpy
scikit-learn
matplotlib
