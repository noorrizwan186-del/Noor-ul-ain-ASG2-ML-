# Assignment Report: Full-Stack Machine Learning App

## Dataset
The project uses the provided Pakistan Pharmaceutical Products and Pricing spreadsheet. It contains 1,630 original rows and seven source columns: Name, Company, Price_before, Discount, Price_After, Pack_Size, and Availability. Exact duplicate rows are removed before model training.

## Model 1: Linear Regression
- **Target:** `Price_After` (final medicine price in PKR).
- **Features:** medicine name, company, numeric original price, discount percentage, and pack size.
- **Evaluation:** R² = 1.000; mean absolute error = PKR 1.13 on the held-out test set in this run.
- **Interpretation:** the very high R² is expected because the final price is strongly determined by the original price and discount in this dataset. It should not be treated as proof of real-world performance on new sources of data.

## Model 2: Decision Tree Classification
- **Target:** `Availability` with only the clear labels `Available` and `Sold Out`.
- **Why filtering is needed:** the source Availability column contains mixed/incorrect values that resemble pack sizes, so other labels are excluded rather than guessed.
- **Features:** medicine name, company, numeric original price, discount percentage, pack size, and final price.
- **Evaluation:** accuracy = 0.644 (64.4%) on the held-out test set in this run.
- **Interpretation:** the model's availability predictions are limited; the score is moderate and should be improved with a cleaner, more reliable availability dataset.

## Web app
The FastAPI backend loads the saved pipelines and provides POST endpoints `/predict/regression` and `/predict/classification`. The HTML page contains a form for each endpoint. The app is intended as a classroom demonstration, not for actual medicine decisions.

## How to run
1. Install packages: `python -m pip install -r requirements.txt`
2. Train models: `python train_models.py`
3. Start app: `python -m uvicorn main:app --reload`
4. Open `http://127.0.0.1:8000`

Scores may vary if the dataset or training procedure is changed.
