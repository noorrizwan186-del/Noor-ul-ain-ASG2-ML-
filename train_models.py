# Train and evaluate both machine learning models
# Run this file after installing requirements.txt.

import os
import re
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, classification_report

DATA_FILE = "data/pharmaceutical_data.xlsx"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load the original spreadsheet.
df = pd.read_excel(DATA_FILE)

# Convert the original price text, such as "Rs 202", into a number.
df["Price_before_numeric"] = pd.to_numeric(
    df["Price_before"].astype(str).str.replace("Rs", "", regex=False).str.replace(",", "", regex=False).str.strip(),
    errors="coerce"
)

# Convert a discount such as "10% Off" into the number 10.
df["Discount_percent"] = pd.to_numeric(
    df["Discount"].astype(str).str.extract(r"(\d+(?:\.\d+)?)")[0],
    errors="coerce"
).fillna(0)

df["Price_After"] = pd.to_numeric(df["Price_After"], errors="coerce")
for col in ["Name", "Company", "Pack_Size"]:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

# Remove exact duplicate rows so repeated records do not dominate the training.
df = df.drop_duplicates().copy()

# ---------------- REGRESSION: predict the final medicine price ----------------
reg_features = ["Name", "Company", "Price_before_numeric", "Discount_percent", "Pack_Size"]
reg_data = df[reg_features + ["Price_After"]].dropna(subset=["Price_After"]).copy()
X_reg = reg_data[reg_features]
y_reg = reg_data["Price_After"]

reg_categorical = ["Name", "Company", "Pack_Size"]
reg_numeric = ["Price_before_numeric", "Discount_percent"]

reg_preprocess = ColumnTransformer([
    ("categories", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), reg_categorical),
    ("numbers", Pipeline([
        ("fill", SimpleImputer(strategy="median"))
    ]), reg_numeric)
])

reg_model = Pipeline([
    ("preprocess", reg_preprocess),
    ("model", LinearRegression())
])

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
reg_model.fit(X_train, y_train)
reg_predictions = reg_model.predict(X_test)

print("REGRESSION RESULTS")
print("Rows used:", len(reg_data))
print("R² score:", round(r2_score(y_test, reg_predictions), 3))
print("Mean absolute error (PKR):", round(mean_absolute_error(y_test, reg_predictions), 2))

joblib.dump(reg_model, os.path.join(MODEL_DIR, "regression_model.pkl"))
print("Saved: models/regression_model.pkl")

# ---------------- CLASSIFICATION: predict Available or Sold Out ----------------
# In this spreadsheet, some Availability cells contain pack sizes or other text.
# Keep only the two clear labels so the classification target is reliable.
df["Availability_clean"] = df["Availability"].astype(str).str.strip().str.lower()
valid_labels = {"available": "Available", "sold out": "Sold Out"}
df["Availability_clean"] = df["Availability_clean"].map(valid_labels)

class_features = ["Name", "Company", "Price_before_numeric", "Discount_percent", "Pack_Size", "Price_After"]
class_data = df[class_features + ["Availability_clean"]].dropna(subset=["Availability_clean"]).copy()

# Keep only rows with a valid final price too, so missing values are easier to explain.
class_data = class_data.dropna(subset=["Price_After"])
X_class = class_data[class_features]
y_class = class_data["Availability_clean"]

class_categorical = ["Name", "Company", "Pack_Size"]
class_numeric = ["Price_before_numeric", "Discount_percent", "Price_After"]

class_preprocess = ColumnTransformer([
    ("categories", Pipeline([
        ("fill", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ]), class_categorical),
    ("numbers", Pipeline([
        ("fill", SimpleImputer(strategy="median"))
    ]), class_numeric)
])

class_model = Pipeline([
    ("preprocess", class_preprocess),
    ("model", DecisionTreeClassifier(max_depth=8, random_state=42, class_weight="balanced"))
])

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)
class_model.fit(X_train_c, y_train_c)
class_predictions = class_model.predict(X_test_c)

print("\nCLASSIFICATION RESULTS")
print("Rows used:", len(class_data))
print("Class counts:")
print(y_class.value_counts())
print("Accuracy:", round(accuracy_score(y_test_c, class_predictions), 3))
print("\nClassification report:")
print(classification_report(y_test_c, class_predictions, zero_division=0))

joblib.dump(class_model, os.path.join(MODEL_DIR, "classification_model.pkl"))
print("Saved: models/classification_model.pkl")

# Save feature names and a small note for the API and future maintenance.
joblib.dump(reg_features, os.path.join(MODEL_DIR, "regression_features.pkl"))
joblib.dump(class_features, os.path.join(MODEL_DIR, "classification_features.pkl"))
print("\nDone. Both trained pipelines and feature lists are saved in the models folder.")
