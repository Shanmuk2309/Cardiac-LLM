# Cardiac-LLM: AI-Powered Cardiac Risk Assessment

A machine learning project that predicts cardiac risk using clinical data, provides SHAP-based explanations, and generates patient-friendly reports using large language models (LLMs).

## Features

- **Risk Prediction**: Uses Random Forest model trained on heart disease datasets to predict cardiac risk probability.
- **Explainability**: Employs SHAP (SHapley Additive exPlanations) to identify key factors influencing predictions.
- **LLM Reports**: Generates detailed, easy-to-understand reports for patients using OpenAI GPT (or other LLMs).
- **Data Visualization**: Includes radar charts, risk gauges, and other plots for better insights.
- **Modular Code**: Separate scripts for training, prediction, and report generation.

## Installation

### Prerequisites
- Python 3.8 or higher (tested on 3.14)
- Git (for cloning the repository)

### Setup Steps

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd Cardiac-LLM
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key** (for LLM reports):
   - Obtain an API key from [OpenAI Platform](https://platform.openai.com/api-keys).
   - Create a `.env` file in the root directory:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```
   - If using a different LLM, update `llm/generate_report.py` accordingly.

## Usage

### Training the Model
Run the training script to build the XGBoost model:
```bash
python scripts/train_model.py
```
This will train the model on the provided dataset and save it to the `models/` directory.

### Prediction and Explanation
Run the main script to predict risk and generate reports:
```bash
python scripts/explain_and_predict.py
```
- Loads the trained model.
- Processes sample patient data.
- Computes SHAP values for explainability.
- Generates visualizations (saved to `plots/`).
- Calls the LLM to create a patient report.

### Sample Output
- Risk Probability: e.g., 75.2%
- Top SHAP Drivers: Key features like age, cholesterol, etc.
- Patient Report: A detailed, simple-language explanation from the LLM.

## Project Structure

```
Cardiac-LLM/
├── data/
│   └── heart_statlog_cleveland_hungary_final.csv  # Dataset
├── llm/
│   ├── __init__.py
│   └── generate_report.py  # LLM report generation
├── models/  # Trained models
├── plots/  # Generated visualizations
├── scripts/
│   ├── explain_and_predict.py  # Main prediction script
│   └── train_model.py  # Model training script
├── .env  # Environment variables (API keys)
├── .gitignore  # Ignore sensitive files
├── requirements.txt  # Python dependencies
└── README.md  # This file
```

## Dependencies

- `python-dotenv==1.0.0`: Environment variable management
- `pandas`: Data manipulation
- `numpy`: Numerical computations
- `scikit-learn`: Machine learning utilities
- `shap`: Explainable AI
- `joblib`: Model serialization
- `matplotlib`: Plotting
- `openai`: LLM API client

## Data

The project uses the "Heart Disease" dataset from the Cleveland and Hungarian datasets, available at [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease).

- **Source**: `data/heart_statlog_cleveland_hungary_final.csv`
- **Features**: Age, sex, chest pain type, resting blood pressure, cholesterol, etc.
- **Target**: Presence of heart disease (0 or 1)

## Model

- **Algorithm**: Random Forest Classifier
- **Training**: Performed in `scripts/train_model.py`
- **Evaluation**: Accuracy, precision, recall, etc. (check script output)
- **Saved Model**: `models/heart_model.pkl`

## Configuration

- **API Keys**: Stored in `.env` for security.
- **Model Path**: Update paths in scripts if changed.
- **LLM Model**: Currently set to `gpt-4o-mini`. Change in `llm/generate_report.py` if needed.

## Troubleshooting

- **API Key Errors**: Ensure `.env` has a valid `OPENAI_API_KEY`. Check credits at OpenAI.
- **Import Errors**: Activate the virtual environment and reinstall dependencies.
- **Model Not Found**: Run `train_model.py` first.
- **LLM Quota Exceeded**: Add credits to your OpenAI account or switch to a free tier/local LLM.

## Acknowledgments

- Dataset: UCI Machine Learning Repository
- Libraries: XGBoost, SHAP, OpenAI
- Inspiration: Explainable AI in healthcare