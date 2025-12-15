import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env file
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY missing! Check your .env file.")

# Configure the library
genai.configure(api_key=API_KEY)

def generate_cardiac_report(result):
    # Build prompt
    prompt = f"""
persona - You are a medical assistant who is an expert in cardio related issues. 

context - You have been provided with the analysis of a patient's cardiac risk based on their clinical data. Using that data, you need to explain 
          it to the patient so that they can understand their risk factors and take necessary actions.
        Risk Probability: {result['risk_probability']*100:.2f}%
        Clinical Summary: {result['clinical_summary']}
        A total of {len(result['top_shap_drivers'])} features influenced this risk score. 
        
Task - Based on this information provided, you need to generate a report which explains the cardiac risk to the patient in simple terms.
       The report should include a detail explanation of the patient's cardiac risk, key factors driving this risk, and the reccommendations
       that the patient should follow to reduce this risk.

Format - A paragraph content of the explaination of the pateint's cardiac risk. Use simple language that patients can understand and include necessary details.
         With a heading of key drivers of risk factors, list the top 5 features that influenced the risk score and why they are influencing.
         With a heading of recommendations, list 5 reccommendations that the patient should follow to reduce this risk and use numbering.
         Don't use technical jargon, keep it simple and easy to understand.
         Don't use markdown formatting.

Top SHAP Drivers:
"""
    for d in result["top_shap_drivers"]:
        prompt += f"\n - {d['feature']}: value={d['value']}, shap={d['shap_value']}"

    try:
        # --- UPDATE: Using the available model from your list ---
        model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")
        
        response = model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return f"Error generating report: {str(e)}"