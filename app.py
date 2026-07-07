import streamlit as st
import pandas as pd
import joblib
import os
import csv
from datetime import datetime
from groq import Groq

st.set_page_config(page_title="Catalog Defect Detector", layout="centered")

# Load the trained model once
model = joblib.load("xgb_model.pkl")
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

st.title("Catalog Defect Detection + Attribute Correction")
st.write("Enter a product listing to check if it's likely defective, and generate a missing description if needed.")

# --- Input form ---
title = st.text_input("Product Title")
category = st.text_input("Category")
has_price = st.checkbox("Has a price listed?", value=True)
rating_number = st.number_input("Number of ratings", min_value=0, value=0)
average_rating = st.number_input("Average rating", min_value=0.0, max_value=5.0, value=0.0)
store = st.text_input("Store/Seller name", value="Unknown")

if st.button("Check Listing"):
    if not title or not category:
        st.warning("Please fill in at least Title and Category.")
    else:
        # Build feature row matching training data structure
        title_length = len(title.split())
        # Simple encodings - in production you'd reuse the original category mappings,
        # here we use a placeholder since new stores/categories won't exist in training data
        store_encoded = abs(hash(store)) % 1000
        category_encoded = abs(hash(category)) % 100

        features_row = pd.DataFrame([{
            'title_length': title_length,
            'has_price': has_price,
            'average_rating': average_rating,
            'rating_number': rating_number,
            'store_encoded': store_encoded,
            'category_encoded': category_encoded
        }])

        # Predict probability, apply our chosen threshold of 0.4
        prob = model.predict_proba(features_row)[0][1]
        is_defective = prob >= 0.4

        st.subheader("Result")
        st.write(f"Defect probability: **{prob:.2f}**")

        generated_description = None

        if is_defective:
            st.error("⚠️ This listing is likely DEFECTIVE (missing key attributes)")

            # Generate description using zero-shot (our winning strategy from Stage 3)
            prompt = f"""Write a short product description (2-3 sentences) for this item.

Title: {title}
Category: {category}
Features: Not available

Description:"""

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150
            )
            generated_description = response.choices[0].message.content.strip()

            st.subheader("Generated Description (via LLM)")
            st.write(generated_description)
        else:
            st.success("✅ This listing looks complete.")

        # --- Logging for monitoring ---
        log_file = "monitoring_log.csv"
        log_exists = os.path.isfile(log_file)

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if not log_exists:
                writer.writerow(["timestamp", "title", "category", "defect_probability", "is_defective", "generated_description"])
            writer.writerow([
                datetime.now().isoformat(),
                title,
                category,
                round(prob, 3),
                is_defective,
                generated_description if generated_description else ""
            ])

        st.caption("This check has been logged for monitoring.")
