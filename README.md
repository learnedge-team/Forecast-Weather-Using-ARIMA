# Forecast-Weather-Using-ARIMA

Deployed link : https://learnedge-forecast-weather-using-arima.streamlit.app/


A Streamlit dashboard for daily temperature forecasting using an ARIMA model trained on the Muthuj7 weather dataset (Kaggle). This repository includes training and inference code, model artifacts, and interactive visualizations.

Features
- Train an ARIMA model and save artifacts (`trainmodel.py`).
- Interactive Streamlit dashboard (`app.py`) with:
  - Forecast generation (configurable days)
  - Historical analysis and monthly/yearly views
  - Model evaluation (MAE/RMSE/MAPE, residuals)
  - Data downloads (forecast & historical)
- Additional weather feature plotting if available (humidity, wind speed, pressure)

Quickstart
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. Train the model (downloads dataset via kagglehub and saves artifacts):

```bash
python trainmodel.py
```

This creates `weather_arima_model.pkl` and `historical_weather_data.pkl` in the repo folder.

3. Run the Streamlit dashboard:

```bash
streamlit run app.py
```

Files
- `app.py` — Streamlit dashboard.
- `trainmodel.py` — Training pipeline: download, preprocess, train ARIMA, save artifacts.
- `weather_arima_model.pkl` — Saved model and artifacts produced by training.
- `historical_weather_data.pkl` — Saved historical data snapshot.
- `requirements.txt` — Python dependencies.

Notes & Troubleshooting
- If the dataset download fails, ensure you have Kaggle credentials or change `trainmodel.py` to use a local CSV.
- If Streamlit reports deprecation warnings, update calls to `st.plotly_chart(..., width='stretch')` and `st.dataframe(..., width='stretch')` (already applied in this repo).

License
- Add a license file as needed (e.g., MIT). See `CONTRIBUTING.md` for guidance.

Contact
- For questions or issues, open an issue in this repository or reach out to the maintainer.

