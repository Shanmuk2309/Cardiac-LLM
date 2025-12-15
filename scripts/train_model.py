import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import shap
import os

# 1. Load the ORIGINAL DATASET (CSV)
df = pd.read_csv("data/heart_statlog_cleveland_hungary_final.csv")

print("Loaded dataset with shape:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Removing the data which contains invalid values
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric, errors="ignore")
df = df.dropna()

# 3. Split features + target
y = df["target"]
X = df.drop(columns=["target"])

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Training RandomForest model
model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# 7. Evaluation
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\n=== Model Evaluation ===")
print("AUROC:", roc_auc_score(y_test, y_pred_proba))
print(classification_report(y_test, (y_pred_proba > 0.5).astype(int)))


# 8. Save model
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/heart_model.pkl")
print("\nModel saved at: models/heart_model.pkl")

# 9. Create SHAP Explainer
# sample 100 rows for background (speeds up SHAP)
background = X_train.sample(min(100, len(X_train)), random_state=42)

explainer = shap.TreeExplainer(model, data=background)

joblib.dump(explainer, "models/shap_explainer.pkl")
print("SHAP explainer saved at: models/shap_explainer.pkl")

# 10. Save the feature names (needed for prediction script)
joblib.dump(list(X.columns), "models/feature_names.pkl")
print("Feature names saved at: models/feature_names.pkl")


print("\nTraining completed successfully!")
