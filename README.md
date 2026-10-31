# Pharmaceutical Full-Stack Machine Learning App

This project uses the supplied **Pakistan Pharmaceutical Products and Pricing** spreadsheet.

## What the app does

1. **Regression:** predicts `Price_After` (final price in PKR) using medicine name, company, original price, discount percentage, and pack size.
2. **Classification:** predicts `Available` or `Sold Out`. The source `Availability` column contains some incorrect/mixed entries (including pack sizes), so the model uses only rows with the two clear labels.

The regression model is Linear Regression. The classification model is a Decision Tree Classifier. Both complete pipelines are saved with `joblib`, including preprocessing.

## Files

- `data/pharmaceutical_data.xlsx`: original supplied dataset.
- `data/pharmaceutical_data_cleaned.csv`: additional columns converted to numeric/cleaned labels for inspection.
- `train_models.py`: trains, evaluates, and saves both models.
- `main.py`: FastAPI backend and POST endpoints.
- `static/index.html`: two prediction forms.
- `models/`: trained models are created here after training.
- `pharmaceutical_ml_assignment.ipynb`: beginner-friendly notebook with the same steps.

## Run on Windows

Open Command Prompt or the VS Code terminal in this project folder.

### 1. Install packages

```bash
python -m pip install -r requirements.txt
```

### 2. Train and save the models

```bash
python train_models.py
```

This prints the R² score, mean absolute error, classification accuracy, and a classification report. It also creates the model files in `models/`.

### 3. Start the website and API

```bash
python -m uvicorn main:app --reload
```

Open **http://127.0.0.1:8000** in your browser. The backend API documentation is at **http://127.0.0.1:8000/docs**.

Keep the terminal running while using the website. Press `Ctrl+C` to stop the server.

## Run the Jupyter notebook

```bash
python -m pip install notebook
python -m notebook
```

Open `pharmaceutical_ml_assignment.ipynb` in Jupyter. Run cells from top to bottom. The notebook trains the models and saves them into `models/`.

## Important notes

- Exact duplicate rows are removed before training.
- Missing discounts are treated as 0% for this project; confirm this assumption with your instructor if needed.
- The availability field is noisy, so invalid labels are excluded instead of guessed.
- Results depend on the quality and size of the dataset. Do not claim scores before running the notebook/script.
- This is a student learning demo, not a reliable real-world medicine availability or pricing service.
- The assignment asks for a GitHub repository and a short README. Upload the project files to your own repository after testing them.
