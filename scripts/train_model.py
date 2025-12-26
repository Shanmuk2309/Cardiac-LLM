import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
import joblib
import shap
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc

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

# ... existing code ...

# 11. Generate Performance Visualizations
print("\nGenerating performance plots...")
os.makedirs("plots", exist_ok=True)

# --- Plot A: Confusion Matrix ---
y_pred = (y_pred_proba > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Heart Disease"])

plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: Cardiac Risk Model")
plt.savefig("plots/confusion_matrix.png")
plt.close()
print("Saved: plots/confusion_matrix.png")

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
plt.savefig("plots/roc_curve.png")
plt.close()
print("Saved: plots/roc_curve.png")

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
plt.savefig("plots/feature_importance.png")
plt.close()
print("Saved: plots/feature_importance.png")
