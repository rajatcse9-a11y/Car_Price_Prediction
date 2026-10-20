# 🚗 Car Price Prediction

A Machine Learning web application that predicts the estimated selling price of a car based on its features, such as brand, year, fuel type, transmission, kilometers driven, and other relevant attributes.

## 📌 Project Overview

The **Car Price Prediction** project uses Machine Learning to estimate car prices based on historical vehicle data. The application provides an interactive Streamlit interface where users can enter car details and get a predicted price.

## ✨ Features

* Predicts estimated car selling prices using a trained ML model.
* Interactive and user-friendly Streamlit interface.
* Handles categorical features using saved encoders.
* Uses a saved model and preprocessing-related files for predictions.
* Displays predicted prices in Indian Rupees (₹).
* Supports a structured, reusable prediction workflow.

## 🛠️ Tech Stack

* **Language:** Python
* **Machine Learning:** Scikit-learn
* **Data Processing:** Pandas, NumPy
* **Web Application:** Streamlit
* **Model Persistence:** Pickle
* **Version Control:** Git and GitHub

## 📂 Project Structure

```text
Car-Price-Prediction/
├── app.py
├── car_price_model.pkl
├── encoders.pkl
├── cat_cols.pkl
├── feature_columns.pkl
├── numeric_ranges.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: Adjust the filenames above to match your actual repository structure.*

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/rajatcse9-a11y/Car_Price_Prediction.git
cd Car-Price-Prediction
```

### 2. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 🔍 How It Works

1. The user enters the car's details in the Streamlit interface.
2. The application validates and prepares the input features.
3. Saved encoders and feature information transform the input into the expected format.
4. The trained Machine Learning model predicts the estimated car price.
5. The predicted price is displayed to the user in Indian Rupees.

## 📈 Model and Prediction

The application loads a previously trained model from a `.pkl` file. The saved model and supporting preprocessing files are reused during prediction, so retraining is not required every time the application starts.

## 🚀 Future Improvements

* Deploy the application online.
* Improve prediction accuracy through model evaluation and hyperparameter tuning.
* Add interactive visualizations and feature insights.
* Expand support for different car brands and configurations.

## 👨‍💻 Author

**Rajat Maurya**

* GitHub: [rajatcse9-a11y](https://github.com/rajatcse9-a11y)
* LinkedIn: [Rajat Maurya](https://www.linkedin.com/in/rajat-maurya-6a6b72354/)

---

*This project was developed for practical learning and demonstration of Machine Learning model deployment using Streamlit.*
