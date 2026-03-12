# Intelligent Learning Recommendation System

## Overview
The Intelligent Learning Recommendation System is an end-to-end machine learning application that predicts a student’s knowledge level and recommends relevant learning resources. The system integrates data preprocessing, feature engineering, supervised learning, and recommendation logic into a deployable Streamlit-based web application.

This project is designed as a production-ready academic and portfolio system, demonstrating practical ML workflow design, deployment readiness, and user-facing interface development.

---

## Features

### Knowledge Level Prediction
- Uses an XGBoost classification model.
- Predicts student knowledge level as:
  - Beginner  
  - Intermediate  
  - Advanced  
- Built using engineered student engagement features.

### Resource Recommendation System
- Recommends relevant learning resources from the dataset.
- Uses a lightweight hybrid-safe recommender suitable for deployment.
- Avoids large data dependencies while maintaining meaningful suggestions.

### Interactive Web Interface
- Built using Streamlit.
- Allows user to:
  - Select student ID  
  - Predict knowledge level  
  - View recommended learning resources  
  - Inspect student data

---

## Machine Learning Pipeline

### Data Processing
- Cleaned and merged multiple educational datasets.
- Removed missing values and duplicates.
- Generated meaningful engagement-based features.

### Feature Engineering
- Engagement score generation.
- Categorical encoding using one-hot encoding.
- Knowledge level classification target derived from scores.

### Model Training
- XGBoost classifier trained on engineered dataset.
- Evaluated using accuracy, precision, recall, and F1-score.

### Recommendation Logic
- Original SVD-based system adapted for deployment safety.
- Final recommender:
  - Uses metadata from `vle.csv`
  - Applies popularity-based ranking
  - Provides deterministic and stable outputs.

---

## Installation

Install dependencies:
pip install -r requirements.txt


---

## Running Locally

Run the Streamlit app:
streamlit run src/app/app.py

The app will open in your browser at:
http://localhost:8501


---

## Deployment

The application is deployed on Render using a Python web service configuration.

### Render Configuration

**Build Command**
pip install -r requirements.txt
**Start Command**
streamlit run src/app/app.py --server.port $PORT --server.address 0.0.0.0

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Joblib  

---

## Future Improvements

- Reintroduce advanced hybrid recommender (SVD + content filtering).
- Add student cold-start personalization.
- Improve UI with richer analytics dashboards.
- Enable user-uploaded datasets.

---

## Live Demo

https://intelligent-learning-recommendation.onrender.com/
