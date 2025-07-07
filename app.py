import gradio as gr
import cv2
import numpy as np
import pandas as pd
import pickle
from PIL import Image
import google.generativeai as genai

# --- Gemini API setup ---
GEMINI_API_KEY = "AIzaSyCDkWv9Lw5FiIXTbjfZ4Q26VveoSeQvSgM"  # Replace with a secure method in production
genai.configure(api_key=GEMINI_API_KEY)

# --- Feature extraction and utility functions ---
def segment_soil(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_soil = np.array([10, 30, 30])
    upper_soil = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_soil, upper_soil)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def extract_color_features(image, mask):
    soil_region = cv2.bitwise_and(image, image, mask=mask)
    hsv = cv2.cvtColor(soil_region, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(soil_region, cv2.COLOR_BGR2LAB)
    features = {}
    for i, channel in enumerate(['B', 'G', 'R']):
        features[f'rgb_{channel}_mean'] = np.mean(soil_region[mask > 0][:, i])
        features[f'rgb_{channel}_std'] = np.std(soil_region[mask > 0][:, i])
    for i, channel in enumerate(['H', 'S', 'V']):
        features[f'hsv_{channel}_mean'] = np.mean(hsv[mask > 0][:, i])
        features[f'hsv_{channel}_std'] = np.std(hsv[mask > 0][:, i])
    for i, channel in enumerate(['L', 'a', 'b']):
        features[f'lab_{channel}_mean'] = np.mean(lab[mask > 0][:, i])
        features[f'lab_{channel}_std'] = np.std(lab[mask > 0][:, i])
    return features

def estimate_moisture(image, mask):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]
    l_mean = np.mean(l_channel[mask > 0])
    l_std = np.std(l_channel[mask > 0])
    gloss_score = l_std / 255.0
    darkness_score = (255 - l_mean) / 255.0
    moisture_score = (gloss_score + darkness_score) / 2
    return moisture_score, gloss_score, darkness_score

def moisture_label(moisture_score):
    if moisture_score > 0.7:
        return "High"
    elif moisture_score < 0.4:
        return "Low"
    else:
        return "Moderate"

# --- Load the trained machine learning model ---
with open('model.pkl', 'rb') as f:
    soil_model = pickle.load(f)

# --- Gemini recommendation function (strict and filtered) ---
def get_gemini_recommendations(features, pred, moisture_score, moisture_lvl):
    prompt = f"""
You are a soil expert. Based on the features and predictions below, return exactly 3 to 5 bullet-point recommendations. Each should be clear, **no longer than 25 words**, and **actionable**.

Avoid explanations, introductions, summaries. Only 3–5 clean bullet points.

Features:
{features}

Predicted Soil Quality: {pred}
Estimated Moisture Score: {moisture_score:.2f} ({moisture_lvl})
"""
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(prompt)

    # Filter bullet points only (up to 5)
    lines = response.text.strip().split("\n")
    bullets = [line.strip() for line in lines if line.strip().startswith(("•", "*", "-"))]
    short_bullets = bullets[:5]

    # Fallback if filtering fails
    if not short_bullets:
        return response.text.strip()

    return "<br>".join(short_bullets)

# --- Gradio prediction function ---
def predict_soil_quality(image):
    try:
        # Convert PIL image to OpenCV format
        image = image.convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Segment and extract features
        mask = segment_soil(image)
        color_features = extract_color_features(image, mask)
        moisture_score, gloss_score, darkness_score = estimate_moisture(image, mask)

        color_features['moisture_score'] = moisture_score
        color_features['gloss_score'] = gloss_score
        color_features['darkness_score'] = darkness_score

        # Predict soil quality
        X_test = pd.DataFrame([color_features])
        pred = soil_model.predict(X_test)[0]
        moisture_lvl = moisture_label(moisture_score)

        # Get Gemini recommendations
        gemini_recs = get_gemini_recommendations(color_features, pred, moisture_score, moisture_lvl)

        # Build result HTML
        result = f"<b>Predicted Soil Quality:</b> {pred}<br>"
        result += f"<b>Estimated Moisture Score:</b> {moisture_score:.2f} ({moisture_lvl})<br>"
        result += f"<b>Recommendations:</b><br>{gemini_recs}"

        return result

    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"

# --- Gradio UI setup ---
demo = gr.Interface(
    fn=predict_soil_quality,
    inputs=gr.Image(type="pil", label="Upload Soil Image"),
    outputs=gr.HTML(label="Prediction"),
    title="Soil Moisture & Quality Analyzer",
    description="Upload a soil image to predict its quality and get recommendations."
)

if __name__ == "__main__":
    demo.launch()
