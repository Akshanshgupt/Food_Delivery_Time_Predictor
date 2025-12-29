# Food Delivery Time Prediction (Streamlit)

This project predicts food delivery time (in minutes) using a trained Random Forest Regressor (≈0.79 R²) built from the notebook `Fdt_Prediction_System.ipynb`. A Streamlit app provides an interactive UI to enter delivery details and get instant predictions, with charts and insights.

## Features
- Loads saved model and feature columns: `rf_model.pkl`, `feature_columns.pkl`
- Mirrors notebook preprocessing:
  - Drops `Order_ID`
  - Imputes missing values (median for `Courier_Experience_yrs`, mode for `Weather`, `Traffic_Level`, `Time_of_Day`, `Vehicle_Type`)
  - Label-encodes categoricals per column using categories found in `Food_Delivery_Times.csv`
- Modern UI with sliders/selects and Plotly visualizations
- Prediction history and dataset insights

## Project Files
- `app.py` – Streamlit UI and preprocessing logic
- `Fdt_Prediction_System.ipynb` – Training and evaluation notebook
- `Food_Delivery_Times.csv` – Training dataset (also used to rebuild encoders)
- `rf_model.pkl` – Trained RandomForestRegressor
- `feature_columns.pkl` – Feature order used by the model
- `requirements.txt` – Python dependencies

## Prerequisites
- Python 3.9–3.11 recommended
  

## Setup
1) Create and activate a virtual environment (optional but recommended).

2) Install dependencies:
```
pip install -r requirements.txt
```

3) Run the app:
```
streamlit run app.py
```

The app will open in your browser (default http://localhost:8501).

## How it works
- On startup, the app loads `rf_model.pkl` and `feature_columns.pkl`.
- It reads `Food_Delivery_Times.csv` to fit a separate `LabelEncoder` for each categorical column. This ensures the UI uses the same categories seen during training.
- Your inputs are imputed/encoded exactly like in the notebook, columns are ordered to match `feature_columns.pkl`, and then passed to the model for prediction.
- If you select a category not present in the training CSV, it falls back to the most frequent category for that column.

## Reproducibility notes
- If you retrain or change preprocessing in the notebook, re-export the artifacts (`rf_model.pkl`, `feature_columns.pkl`) and keep the CSV in sync with the new categories.
- If you want to persist true per-column encoders, save and load a dict of encoders instead of rebuilding from the CSV.

## Troubleshooting
- Error: `Failed to load artifacts: ...`
  - Ensure `rf_model.pkl`, `feature_columns.pkl`, and `Food_Delivery_Times.csv` exist in the same folder as `app.py`.
  - Make sure your Python and library versions match or are compatible (use `requirements.txt`).
- Port already in use
  - Run with a different port: `streamlit run app.py --server.port 8502`

## License
This project is for educational/demo purposes. Use at your discretion.


