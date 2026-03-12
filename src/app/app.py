import sys
import os

# -----------------------------
# Fix import path (IMPORTANT)
# -----------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import streamlit as st
import pandas as pd
import joblib

from src.recommender.recommendation_engine import generate_recommendations


# -----------------------------
# Load Model + Data (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("models_saved/xgboost_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv("data/processed/feature_engineered_data.csv")


model = load_model()
df = load_data()


# -----------------------------
# UI Title
# -----------------------------
st.title("🎓 Intelligent Learning Recommendation System")

st.write("Predict student knowledge level + recommend learning resources")


# -----------------------------
# Student Selection
# -----------------------------
student_id = st.selectbox("Select Student ID", df["id_student"].unique())

student_row = df[df["id_student"] == student_id]


# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("📊 Knowledge Prediction")

if st.button("Predict Knowledge Level"):

    # Drop target column
    X = student_row.drop(["knowledge_level"], axis=1)

    # Apply SAME encoding as training
    X = pd.get_dummies(X)

    # Match training columns exactly
    model_cols = model.get_booster().feature_names

    for col in model_cols:
        if col not in X:
            X[col] = 0

    X = X[model_cols]

    # Predict
    pred = model.predict(X)[0]

    # Convert numeric → label
    label_map_reverse = {
        0: "Beginner",
        1: "Intermediate",
        2: "Advanced"
    }

    pred_label = label_map_reverse[int(pred)]

    st.success(f"Predicted Knowledge Level: **{pred_label}**")

# -----------------------------
# Recommendation Section
# -----------------------------
st.subheader("📚 Resource Recommendations")

if st.button("Get Recommendations"):

    recs = generate_recommendations(student_id)

    if len(recs) == 0:
        st.warning("No recommendations available")

    else:
        # Load metadata ONCE
        vle_df = pd.read_csv("./data/raw/vle.csv")

        # FIX datatype issue (IMPORTANT)
        vle_df["id_site"] = vle_df["id_site"].astype(int)

        st.write("Top Recommended Resources:")

        for i, r in enumerate(recs, 1):

            r = int(r)  # ensure match

            resource_info = vle_df[vle_df["id_site"] == r]

            if not resource_info.empty:

                name = resource_info.iloc[0]["activity_type"]
                module = resource_info.iloc[0]["code_module"]

                st.markdown(
                    f"""
                    **{i}. 📘 {name}**  
                    • Resource ID: {r}  
                    • Module: {module}
                    """
                )

            else:
                st.write(f"{i}. Resource ID: {r}")

# -----------------------------
# Student Info View
# -----------------------------
st.subheader("📌 Student Details")

st.dataframe(student_row)