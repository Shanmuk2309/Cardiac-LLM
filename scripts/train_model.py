import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

# Define root directory
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 1. Load the ORIGINAL DATASET (CSV)
CSV_PATH = os.path.join(ROOT_DIR, "data/heart_statlog_cleveland_hungary_final.csv")
df = pd.read_csv(CSV_PATH)

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
MODEL_DIR = os.path.join(ROOT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved at: {MODEL_PATH}")

# 9. Create SHAP Explainer
# sample 100 rows for background (speeds up SHAP)
background = X_train.sample(min(100, len(X_train)), random_state=42)

# Save background data instead of explainer
BACKGROUND_PATH = os.path.join(MODEL_DIR, "shap_background.pkl")
joblib.dump(background, BACKGROUND_PATH)
print(f"SHAP background data saved at: {BACKGROUND_PATH}")

# 10. Save the feature names (needed for prediction script)
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
joblib.dump(list(X.columns), FEATURES_PATH)
print(f"Feature names saved at: {FEATURES_PATH}")


print("\nTraining completed successfully!")

# ... existing code ...

# 11. Generate Performance Visualizations
print("\nGenerating performance plots...")
PLOTS_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# --- Plot A: Confusion Matrix ---
y_pred = (y_pred_proba > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Heart Disease"])

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Cardiac Risk Model")
CONFUSION_PATH = os.path.join(PLOTS_DIR, "confusion_matrix.png")
plt.savefig(CONFUSION_PATH)
plt.close()
print(f"Saved: {CONFUSION_PATH}")

# --- Plot B: ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
ROC_PATH = os.path.join(PLOTS_DIR, "roc_curve.png")
plt.savefig(ROC_PATH)
plt.close()
print(f"Saved: {ROC_PATH}")

# --- Plot C: Global Feature Importance ---
# Get feature importances from the Random Forest
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [X.columns[i] for i in indices]
sorted_importances = importances[indices]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices)), sorted_importances[::-1], align='center')
plt.yticks(range(len(indices)), sorted_features[::-1])
plt.xlabel('Relative Importance')
plt.title('Global Feature Importance (Random Forest)')
FEATURE_PATH = os.path.join(PLOTS_DIR, "feature_importance.png")
plt.savefig(FEATURE_PATH)
plt.close()
print(f"Saved: {FEATURE_PATH}")
