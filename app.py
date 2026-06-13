import streamlit as st
import pickle
import pandas as pd

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Powered Disease Prediction Assistant",
    page_icon="🩺",
    layout="centered"
)

# ---------------- LOAD FILES ----------------
model = pickle.load(open("models/disease_model.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))
symptom_encoder = pickle.load(open("models/symptom_encoder.pkl", "rb"))

desc_df = pd.read_csv("data/symptom_Description.csv")
prec_df = pd.read_csv("data/symptom_precaution.csv")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Project Information")

st.sidebar.info("""
Model: Random Forest

Diseases: 40+

Symptoms: 130+

Accuracy: 98.5%

Developer: Sathwik Naik
""")

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align:center;'>🩺 AI Powered Disease Prediction Assistant</h1>",
    unsafe_allow_html=True
)
st.info(
    "Enter your symptoms below and receive an AI-powered disease prediction with description and recommended precautions."
)
st.markdown("### Select your symptoms and get an instant prediction")

# ---------------- SYMPTOMS ----------------
all_symptoms = sorted(list(symptom_encoder.classes_))

display_symptoms = {
    symptom: symptom.replace("_", " ").title()
    for symptom in all_symptoms
}

selected_display = st.multiselect(
    "Choose Symptoms",
    list(display_symptoms.values())
)

selected_symptoms = [
    k for k, v in display_symptoms.items()
    if v in selected_display
]

# ---------------- PREDICT ----------------
if st.button("🔍 Predict Disease"):

    if not selected_symptoms:
        st.warning("Please select at least one symptom.")

    else:

        input_data = symptom_encoder.transform([selected_symptoms])

        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        disease = label_encoder.inverse_transform(prediction)[0]

        confidence = max(prediction_proba[0]) * 100

        st.success(f"🩺 Predicted Disease: {disease}")

        # Confidence %
        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

        # Confidence Level
        if confidence >= 70:
            st.success(f"🟢 High Confidence Prediction ({confidence:.1f}%)")

        elif confidence >= 40:
            st.warning(f"🟡 Moderate Confidence Prediction ({confidence:.1f}%)")

        else:
            st.error(f"🔴 Low Confidence Prediction ({confidence:.1f}%)")

        # ---------------- DESCRIPTION ----------------
        try:
            description = desc_df[
                desc_df.iloc[:, 0] == disease
            ].iloc[0, 1]

            st.subheader("📖 Disease Description")
            st.info(description)

        except:
            st.warning("Description not found.")

        # ---------------- PRECAUTIONS ----------------
        try:
            precautions = prec_df[
                prec_df.iloc[:, 0] == disease
            ].iloc[0, 1:].dropna().tolist()

            st.subheader("🛡 Recommended Precautions")

            for precaution in precautions:
                st.write(f"✅ {precaution}")

        except:
            st.warning("Precautions not found.")

# ---------------- FOOTER ----------------
st.divider()

st.warning(
    "⚠️ This prediction is for educational purposes only. "
    "Please consult a qualified doctor for professional medical advice."
)
st.divider()

st.caption(
    "Developed by Sathwik Naik | Python |Scikit-Learn | Streamlit"
)