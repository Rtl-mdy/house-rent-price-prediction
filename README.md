# House Rent Price Prediction using Machine Learning

A supervised regression project estimating monthly rent for US apartment listings. The report's selected model is a Random Forest regressor and the Streamlit app exposes bedrooms, bathrooms, square footage, city and state.

## Dataset
- Name: Apartment for Rent Classified (10K)
- Source: UCI Machine Learning Repository, dataset 555
- Source page: https://archive.ics.uci.edu/dataset/555/apartment+for+rent+classified
- Expected local file: `data/apartments_for_rent_classified_10K.csv`
- The source file is semicolon-separated and encoded as `cp1252`.

Download the dataset from the UCI page and place the CSV at the expected path. The dataset is not included in this ZIP.

## Setup (Ubuntu)
```bash
cd House_Rent_Price_Prediction_Project
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python train_model_uci.py
streamlit run app.py
```

## Reproduced report configuration
- Clean rows: 9,917 from 10,000
- Features: bedrooms, bathrooms, square_feet, cityname, state
- Target: monthly `price` in USD
- Split: 80/20, `random_state=42`
- Final model: `RandomForestRegressor(n_estimators=50, min_samples_leaf=2, random_state=42, n_jobs=-1)`
- Reported test metrics: MAE $278.50, RMSE $535.40, R² 0.669
- Output files: cleaned CSV, model comparison CSV, saved joblib pipeline, and MAE chart

Exact values can vary if the downloaded dataset version or installed library behavior differs. The source report's figures are reference results, not a guarantee for a fresh run.

## Important limitations
This is an academic demonstration on historical US listings in USD, not an Indian rent estimator. The source report notes under-prediction for expensive listings and a previously observed State/City dropdown issue; this app places both controls in the same form and rebuilds the city choices from the selected state. Validate the downloaded data and app behavior before submission.
