# 🫀 Heart Disease Severity Prediction (Multiclass Model)

This project aims to classify heart disease severity into five categories—**Normal**, **Mild**, **Moderate**, **Severe**, and **Critical**—using machine learning techniques on the UCI Cleveland dataset. It leverages deep learning, feature engineering, and a Streamlit-based UI for real-time predictions.

## 🚀 Features

- Multiclass classification using a Deep Neural Network (DNN)
- Comparison with traditional models: Logistic Regression, Random Forest, Boosting
- Feature selection and scaling using custom preprocessing pipeline
- SMOTE applied for class imbalance handling
- Interactive Streamlit app for live predictions
- Model interpretability via feature importance analysis

## 🧰 Tech Stack

- **Languages**: Python
- **Libraries**: TensorFlow, Scikit-learn, Pandas, Matplotlib
- **Deployment**: Streamlit
- **Model Files**: `.keras`, `.pkl` for preprocessing and feature selection

## 📁 Project Structure

```
├── cleveland.csv                  # Dataset
├── heart_feature_selector.pkl    # Selected features
├── heart_final_scaler.pkl        # Scaler object
├── heart_preprocessor.pkl        # Preprocessing pipeline
├── heart_severity_model.keras    # Trained DNN model
├── model_training_notebook.ipynb # Model training and evaluation
├── streamlit_app.py              # Streamlit UI for predictions
```

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Harishk2508/Heart-disease-prediction-multiclass-model.git
   cd Heart-disease-prediction-multiclass-model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the Streamlit app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Dataset

- Source: [UCI Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- Preprocessed and balanced using SMOTE for multiclass classification

## 🏆 Results

- Achieved **91%+ accuracy** with DNN
- Improved interpretability using feature importance and cross-validation

## 📌 Author

**Harish Kumar**  
🔗 [LinkedIn](https://linkedin.com/in/harish-kumar-kingston-pydev)  
📧 harishkengineer25@gmail.com

---
