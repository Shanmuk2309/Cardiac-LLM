import os, sys
import matplotlib.pyplot as plt
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# scripts/explain_and_predict.py
from llm.generate_report import generate_cardiac_report
import pandas as pd
import joblib
import shap
import numpy as np
import textwrap

# -----------------------------
# Load model artifacts
# -----------------------------
MODEL_PATH = "models/heart_model.pkl"
EXPLAINER_PATH = "models/shap_explainer.pkl"
FEATURES_PATH = "models/feature_names.pkl"
CSV_PATH = "data/heart_statlog_cleveland_hungary_final.csv"  # used only to compute medians for imputing

model = joblib.load(MODEL_PATH)
explainer = joblib.load(EXPLAINER_PATH)
feature_names = joblib.load(FEATURES_PATH)  # list of 11 feature names in training order

# -----------------------------
# Read CSV (compute medians for imputation of user input)
# -----------------------------
df_master = pd.read_csv(CSV_PATH)
df_master = df_master.replace("?", np.nan)
# coerce numeric where possible (safe as training did)
df_master = df_master.apply(pd.to_numeric, errors="ignore")

# For imputing user inputs, compute median per feature (ignores NaN)
medians = df_master[feature_names].median()

# -----------------------------
# Clinical mapping dictionaries & prompt helpers
# -----------------------------
SEX_MAP = {1: "male", 0: "female"}

CP_OPTIONS = {
    1: "typical angina",
    2: "atypical angina",
    3: "non-anginal pain",
    4: "asymptomatic"
}

FBS_MAP = {1: ">120 mg/dl (high)", 0: "<=120 mg/dl (normal)"}

RESTECG_MAP = {
    0: "normal",
    1: "ST-T wave abnormality",
    2: "left ventricular hypertrophy"
}

EXANG_MAP = {1: "yes", 0: "no"}

SLOPE_MAP = {
    1: "upsloping",
    2: "flat",
    3: "downsloping"
}

# Create reverse lookups for validating textual input
REVERSE_LOOKUPS = {
    "sex": {"male": 1, "female": 0},
    "chest pain type": {v.lower(): k for k, v in CP_OPTIONS.items()},
    "fasting blood sugar": {"high": 1, "1": 1, ">120": 1, "normal": 0, "0": 0, "<=120": 0},
    "resting ecg": {v.lower(): k for k, v in RESTECG_MAP.items()},
    "exercise angina": {"yes": 1, "no": 0},
    "ST slope": {v.lower(): k for k, v in SLOPE_MAP.items()}
}

# -----------------------------
# Utility: validate & coerce user-provided values
# -----------------------------
def coerce_feature_value(feature_name, raw_value):
    """
    Convert raw_value (string or numeric) to the numeric form expected by model,
    using REVERSE_LOOKUPS for categorical features.
    """
    if pd.isna(raw_value) or raw_value == "":
        # missing -> will be imputed later
        return np.nan

    # numeric features (approx): age, resting bp s, cholesterol, max heart rate, oldpeak
    numeric_features = {
        "age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"
    }

    if feature_name in numeric_features:
        try:
            return float(raw_value)
        except Exception:
            raise ValueError(f"Feature '{feature_name}' expects a numeric value. Got '{raw_value}'.")

    # sex
    if feature_name == "sex":
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        s = str(raw_value).strip().lower()
        if s in REVERSE_LOOKUPS["sex"]:
            return REVERSE_LOOKUPS["sex"][s]
        raise ValueError(f"sex must be 'male' or 'female' (or 1/0). Got '{raw_value}'.")

    # chest pain type (cp)
    if feature_name == "chest pain type":
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        s = str(raw_value).strip().lower()
        if s in REVERSE_LOOKUPS["chest pain type"]:
            return REVERSE_LOOKUPS["chest pain type"][s]
        # allow entering numeric strings "1","2" etc.
        if s.isdigit():
            v = int(s)
            if v in CP_OPTIONS:
                return v
        raise ValueError(f"chest pain must be one of: {CP_OPTIONS}. Got '{raw_value}'.")

    # fasting blood sugar
    if feature_name == "fasting blood sugar":
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        s = str(raw_value).strip().lower()
        if s in REVERSE_LOOKUPS["fasting blood sugar"]:
            return REVERSE_LOOKUPS["fasting blood sugar"][s]
        raise ValueError("fasting blood sugar must be 1 (>120 mg/dl) or 0 (<=120 mg/dl).")

    # resting ecg
    if feature_name == "resting ecg":
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        s = str(raw_value).strip().lower()
        if s in REVERSE_LOOKUPS["resting ecg"]:
            return REVERSE_LOOKUPS["resting ecg"][s]
        raise ValueError(f"resting ecg must be one of {RESTECG_MAP}. Got '{raw_value}'.")

    # exercise angina
    if feature_name == "exercise angina":
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        s = str(raw_value).strip().lower()
        if s in REVERSE_LOOKUPS["exercise angina"]:
            return REVERSE_LOOKUPS["exercise angina"][s]
        raise ValueError("exercise angina must be 'yes' or 'no' (or 1/0).")

    # ST slope
    if feature_name == "ST slope":
        if isinstance(raw_value, (int, float)):
            return int(raw_value)
        s = str(raw_value).strip().lower()
        if s in REVERSE_LOOKUPS["ST slope"]:
            return REVERSE_LOOKUPS["ST slope"][s]
        if s.isdigit():
            v = int(s)
            if v in SLOPE_MAP:
                return v
        raise ValueError(f"ST slope must be one of {SLOPE_MAP}. Got '{raw_value}'.")

    # fallback: attempt numeric
    try:
        return float(raw_value)
    except Exception:
        raise ValueError(f"Unrecognized value for feature '{feature_name}': '{raw_value}'")


# -----------------------------
# Main predictor: from dictionary
# -----------------------------
def predict_from_dict(user_dict):
    """
    user_dict: dictionary that may contain a subset or all of the required features.
    Keys must match feature_names (exact strings), but we accept flexible keys by lowercasing.
    Returns structured output: probability, top_shap_drivers, clinical_summary.
    """
    # Build input row with same column order as training
    row_values = []
    for feat in feature_names:
        # accept synonyms by lowercasing keys
        matched = None
        for k in user_dict.keys():
            if str(k).strip().lower() == feat.strip().lower():
                matched = k
                break

        raw_val = user_dict.get(matched, np.nan)
        coerced = coerce_feature_value(feat, raw_val)  # may be np.nan
        row_values.append(coerced)

    X_input = pd.DataFrame([row_values], columns=feature_names)

    # Impute missing values using training medians computed earlier
    for col in feature_names:
        if X_input[col].isna().any():
            X_input[col] = X_input[col].fillna(medians[col])

    # Ensure numeric dtypes
    X_input = X_input.astype(float)

    # Prediction
    proba = float(model.predict_proba(X_input)[0, 1])

    # SHAP explanation
    shap_raw = explainer.shap_values(X_input)
    # handle SHAP formats (some TreeExplainer returns array of shape (n_features,2))
    if isinstance(shap_raw, list):
        shap_values_full = shap_raw[1][0] if len(shap_raw) > 1 else shap_raw[0][0]
    else:
        shap_values_full = shap_raw[0]

    # If shap_values_full is 2D ([n_features,2]) pick column 1
    shap_values = None
    arr = np.array(shap_values_full)
    if arr.ndim == 2 and arr.shape[1] == 2:
        shap_values = arr[:, 1]
    else:
        shap_values = arr.flatten()

    # Top drivers (by absolute SHAP for class 1)
    importance = sorted(
        zip(feature_names, shap_values, X_input.values[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    # Build clinical summary using readable mappings (use original user values if available, else medians)
    # For fields where mapping exists, convert coded numeric to readable string.
    raw_display = {}
    for feat, _, val in importance:
        raw_display[feat] = X_input[feat].iloc[0]

    # Full clinical summary (friendly)
    # Use original CSV "raw" style mapping where appropriate (convert numeric codes back to text)
    clinical_summary = {
        "age": int(X_input["age"].iloc[0]),
        "sex": SEX_MAP.get(int(X_input["sex"].iloc[0]), "unknown"),
        "chest_pain_type": CP_OPTIONS.get(int(X_input["chest pain type"].iloc[0]), str(int(X_input["chest pain type"].iloc[0]))),
        "resting_bp": float(X_input["resting bp s"].iloc[0]),
        "cholesterol": float(X_input["cholesterol"].iloc[0]),
        "fasting_blood_sugar": FBS_MAP.get(int(X_input["fasting blood sugar"].iloc[0]), str(int(X_input["fasting blood sugar"].iloc[0]))),
        "resting_ecg": RESTECG_MAP.get(int(X_input["resting ecg"].iloc[0]), str(int(X_input["resting ecg"].iloc[0]))),
        "max_heart_rate": float(X_input["max heart rate"].iloc[0]),
        "exercise_angina": EXANG_MAP.get(int(X_input["exercise angina"].iloc[0]), str(int(X_input["exercise angina"].iloc[0]))),
        "oldpeak": float(X_input["oldpeak"].iloc[0]),
        "ST_slope": SLOPE_MAP.get(int(X_input["ST slope"].iloc[0]), str(int(X_input["ST slope"].iloc[0])))
    }

    result = {
        "risk_probability": proba,
        "top_shap_drivers": [
            {"feature": f, "shap_value": float(s), "value": float(v)} for f, s, v in importance
        ],
        "clinical_summary": clinical_summary
    }

    try:
        # We need to construct a robust Explanation object for plotting
        # explainer.expected_value might be a list (class 0, class 1) or scalar
        base_val = explainer.expected_value
        if isinstance(base_val, list) or isinstance(base_val, np.ndarray):
            base_val = base_val[1] # Class 1 (Disease) baseline

        # Create the explanation object specifically for this single patient
        explanation = shap.Explanation(
            values=shap_values,           # The calculated SHAP values for this person
            base_values=base_val,         # The average risk (baseline)
            data=X_input.iloc[0].values,  # The patient's actual feature values
            feature_names=feature_names   # The list of column names
        )

        plt.figure(figsize=(10, 6))
        # Create the plot (show=False allows us to save it)
        shap.plots.waterfall(explanation, show=False, max_display=10)
        
        # Save the plot
        save_path = "patient_risk_explanation.png"
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"\n[INFO] Visual explanation saved to: {os.path.abspath(save_path)}")
        
    except Exception as e:
        print(f"[WARNING] Could not generate SHAP plot: {e}")


    # ---------------------------------------------------------
    # VISUALIZATION 2: Clinical Radar Chart (Patient vs. Average Healthy)
    # ---------------------------------------------------------
    try:
        # 1. Select numeric features for the chart
        radar_feats = ["age", "resting bp s", "cholesterol", "max heart rate", "oldpeak"]
        
        # 2. Get averages for "Healthy" (target=0) from the master dataset
        # (df_master is already loaded globally in your script)
        healthy_means = df_master[df_master["target"] == 0][radar_feats].mean()
        
        # 3. Normalize data (0-1 scaling) so different units fit on one chart
        # We use the dataset max values to scale
        max_vals = df_master[radar_feats].max()
        
        patient_vals_norm = []
        healthy_vals_norm = []
        
        for f in radar_feats:
            val = X_input[f].iloc[0]
            # Clip to max to avoid drawing outside the circle
            norm_val = min(val / max_vals[f], 1.0) 
            patient_vals_norm.append(norm_val)
            healthy_vals_norm.append(healthy_means[f] / max_vals[f])
            
        # 4. Prepare plot data (close the loop for radar chart)
        angles = np.linspace(0, 2 * np.pi, len(radar_feats), endpoint=False).tolist()
        patient_vals_norm += [patient_vals_norm[0]]
        healthy_vals_norm += [healthy_vals_norm[0]]
        angles += [angles[0]]
        
        # 5. Plot
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        
        # Draw Healthy Baseline
        ax.plot(angles, healthy_vals_norm, color='green', linewidth=2, linestyle='dashed', label='Avg Healthy Profile')
        ax.fill(angles, healthy_vals_norm, color='green', alpha=0.1)
        
        # Draw Patient
        ax.plot(angles, patient_vals_norm, color='red', linewidth=2, label='Patient')
        ax.fill(angles, patient_vals_norm, color='red', alpha=0.25)
        
        # Labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(radar_feats)
        ax.set_yticklabels([]) # Hide radial numbers
        ax.set_title("Patient Vitals vs. Healthy Baseline", y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        plt.savefig("patient_radar_chart.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[INFO] Radar chart saved to: {os.path.abspath('patient_radar_chart.png')}")

    except Exception as e:
        print(f"[WARNING] Could not generate Radar Chart: {e}")

    # [INSERT THIS BLOCK AFTER THE RADAR CHART]
    # ---------------------------------------------------------
    # VISUALIZATION 3: Risk Gauge Chart
    # ---------------------------------------------------------
    try:
        # Simple Donut Chart acting as a Gauge
        risk_score = proba # This is your calculated probability (0.0 to 1.0)
        
        # Colors: Green (Low), Yellow (Med), Red (High)
        colors = ['#4CAF50', '#FFC107', '#F44336']
        
        plt.figure(figsize=(6, 4))
        
        # Create a half-donut
        # Sizes correspond to Low (0-0.33), Medium (0.33-0.66), High (0.66-1.0)
        plt.pie([1, 1, 1], colors=colors, startangle=0, counterclock=False, 
                wedgeprops=dict(width=0.4, edgecolor='w'), labels=["High", "Med", "Low"], 
                labeldistance=1.1, textprops={'fontsize': 10})
        
        # Draw a circle in the center to make it a donut (and hide the bottom half)
        # Actually, standard pie chart is 360. To make a gauge, we mask the bottom.
        # Simpler approach: Just a horizontal bar with a marker.
        
        # --- Simpler Linear Gauge Approach ---
        plt.clf() # clear previous attempts
        fig, ax = plt.subplots(figsize=(6, 2))
        
        # Draw the gradient bar
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap('RdYlGn_r'), extent=[0, 1, 0, 1])
        
        # Place the marker for patient
        ax.plot([risk_score, risk_score], [0, 1], color='black', linewidth=4)
        ax.text(risk_score, 1.2, f"{risk_score*100:.1f}%", horizontalalignment='center', fontsize=12, fontweight='bold')
        
        # Labels
        ax.set_yticks([])
        ax.set_xticks([0, 0.33, 0.66, 1])
        ax.set_xticklabels(["Low", "Medium", "High", "Critical"])
        ax.set_title("Cardiac Risk Probability", fontsize=12)
        
        plt.savefig("patient_risk_gauge.png", bbox_inches='tight', dpi=150)
        plt.close()
        print(f"[INFO] Risk Gauge saved to: {os.path.abspath('patient_risk_gauge.png')}")

    except Exception as e:
        print(f"[WARNING] Could not generate Gauge Chart: {e}")

    return result


# -----------------------------
# Interactive CLI mode
# -----------------------------
def interactive_prompt():
    print("\nCardiac Risk Predictor — interactive input\n")
    print("Enter values for the following features. Press Enter to use a reasonable default (median).")
    print("If a feature is categorical, some choices are shown.\n")

    user_inputs = {}
    for feat in feature_names:
        prompt = f"{feat} "

        # Show choices for categorical features
        if feat == "sex":
            prompt += "(male/female) "
        elif feat == "chest pain type":
            prompt += f"({', '.join([f'{k}:{v}' for k, v in CP_OPTIONS.items()])}) "
        elif feat == "fasting blood sugar":
            prompt += "(1 if >120 mg/dl else 0) "
        elif feat == "resting ecg":
            prompt += f"({', '.join([f'{k}:{v}' for k, v in RESTECG_MAP.items()])}) "
        elif feat == "exercise angina":
            prompt += "(yes/no) "
        elif feat == "ST slope":
            prompt += f"({', '.join([f'{k}:{v}' for k, v in SLOPE_MAP.items()])}) "

        prompt += f"[median={medians[feat]}]: "

        raw = input(prompt).strip()
        if raw == "":
            user_inputs[feat] = medians[feat]
        else:
            try:
                coerced = coerce_feature_value(feat, raw)
                user_inputs[feat] = coerced
            except Exception as e:
                print(f"Invalid input for {feat}: {e}. Using median {medians[feat]}")
                user_inputs[feat] = medians[feat]

    print("\nComputing prediction ...\n")
    res = predict_from_dict(user_inputs)

    # Pretty print result
    prob = res["risk_probability"]
    print(f"Estimated probability of heart disease (class 1): {prob:.2%}\n")

    print("Top contributing features (feature, shap contribution to risk, input value):")
    for d in res["top_shap_drivers"]:
        print(f" - {d['feature']}: shap={d['shap_value']:.4f}, value={d['value']}")

    print("\nClinical summary:")
    for k, v in res["clinical_summary"].items():
        print(f"  {k}: {v}")

    print("\nIMPORTANT: This is a decision-support estimate. Confirm with clinical evaluation.\n")

    # Generate LLM-based report - Gemini API call
    report = generate_cardiac_report(res)

    print("\n=== LLM-GENERATED CARDIAC REPORT ===\n")
    print(report)

# -----------------------------
# If script run directly, launch interactive prompt
# -----------------------------
if __name__ == "__main__":
    # Provide short instructions
    print(textwrap.dedent("""
    ===== Cardiac LLM Project — Explain & Predict =====
    This script will ask you for feature values, impute any missing values using training medians,
    run the trained RandomForest model, and provide SHAP-based explanatory drivers.
    """))
    interactive_prompt()
