import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("heart_severity_model.keras")
    preprocessor = joblib.load("heart_preprocessor.pkl")
    selector = joblib.load("heart_feature_selector.pkl")
    scaler = joblib.load("heart_final_scaler.pkl")
    return model, preprocessor, selector, scaler

model, preprocessor, selector, scaler = load_artifacts()

st.title("Heart Disease Severity Prediction")
st.write("Input patient clinical data below:")

def user_input_features():
    age = st.slider("Age", 29, 77, 54)

    sex = st.selectbox("Sex", options=["Female", "Male"])
    sex = 0 if sex == "Female" else 1

    cp_label_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp = st.selectbox("Chest Pain Type", options=list(cp_label_map.keys()))
    cp = cp_label_map[cp]

    trestbps = st.slider("Resting Blood Pressure (mm Hg)", 94, 200, 130)

    chol = st.slider("Cholesterol (mg/dL)", 126, 564, 246)
    # Normalizing cholesterol (range from training data: 126–564)
    # chol = (chol - 126) / (564 - 126)
  # Apply transformation for model to "notice" difference

    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", options=["No", "Yes"])
    fbs = 1 if fbs == "Yes" else 0

    restecg_label_map = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Probable/definite LVH": 2
    }
    restecg = st.selectbox("Resting ECG", options=list(restecg_label_map.keys()))
    restecg = restecg_label_map[restecg]

    thalach = st.slider("Max Heart Rate Achieved", 71, 202, 150)

    exang = st.selectbox("Exercise Induced Angina", options=["No", "Yes"])
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.2, 1.0, step=0.1)

    slope_label_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope = st.selectbox("Slope of Peak Exercise ST", options=list(slope_label_map.keys()))
    slope = slope_label_map[slope]

    ca = st.selectbox("Major Vessels Colored by Fluoroscopy", options=[0, 1, 2, 3, 4])

    thal_label_map = {
        "Normal": 3,
        "Fixed Defect": 6,
        "Reversible Defect": 7
    }
    thal = st.selectbox("Thalassemia Type", options=list(thal_label_map.keys()))
    thal = thal_label_map[thal]

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# Transform for prediction
X_processed = preprocessor.transform(input_df)
X_selected = selector.transform(X_processed)
X_scaled = scaler.transform(X_selected)

prediction_proba = model.predict(X_scaled)[0]
predicted_class = np.argmax(prediction_proba)
confidence = prediction_proba[predicted_class]

severity_dict = {
    0: "Normal",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Critical"
}

st.markdown(f"### Predicted Heart Disease Severity Class: **{predicted_class}**")
st.markdown(f"**Severity Description:** {severity_dict[predicted_class]}")
st.markdown(f"**Confidence:** {confidence:.2%}")

# Sidebar Presets
example_cases = {
    "Normal": {
        "age": 35, "sex": 0, "cp": 0, "trestbps": 110, "chol": 180,
        "fbs": 0, "restecg": 0, "thalach": 170, "exang": 0,
        "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 3
    },
    "Mild": {
        "age": 45, "sex": 0, "cp": 1, "trestbps": 120, "chol": 210,
        "fbs": 0, "restecg": 1, "thalach": 160, "exang": 0,
        "oldpeak": 0.5, "slope": 2, "ca": 0, "thal": 6
    },
    "Moderate": {
        "age": 55, "sex": 1, "cp": 1, "trestbps": 135, "chol": 240,
        "fbs": 1, "restecg": 1, "thalach": 140, "exang": 1,
        "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 6
    },
    "Severe": {
        "age": 60, "sex": 1, "cp": 2, "trestbps": 150, "chol": 300,
        "fbs": 1, "restecg": 2, "thalach": 130, "exang": 1,
        "oldpeak": 2.5, "slope": 1, "ca": 2, "thal": 7
    },
    "Critical": {
        "age": 70, "sex": 1, "cp": 3, "trestbps": 180, "chol": 360,
        "fbs": 1, "restecg": 2, "thalach": 110, "exang": 1,
        "oldpeak": 4.0, "slope": 0, "ca": 3, "thal": 7
    }
}

st.sidebar.title("Load Example Case")
example = st.sidebar.selectbox("Select example to load", [""] + list(example_cases.keys()))
if example:
    example_data = example_cases[example]
    st.sidebar.markdown("### Example feature values (copy & enter manually):")
    st.sidebar.json(example_data)
    st.sidebar.info(
        "Note: Streamlit currently does not support automatic input update. "
        "Please manually update the inputs above to these values for testing."
    )
