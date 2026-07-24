# 🤖 AutoML Model Comparison System

An intelligent Machine Learning application built with **Python** and **Streamlit** that automatically detects the type of machine learning problem, preprocesses data, trains multiple models, performs hyperparameter tuning, compares their performance, and selects the best model automatically.

---

# 📌 Features

✅ Upload your own dataset (CSV)

✅ Automatic data preprocessing

- Missing value handling
- Categorical encoding
- Feature scaling
- Train/Test split

✅ Automatic problem type detection

- Classification
- Regression

(or manually choose the problem type)

✅ Trains multiple Machine Learning algorithms automatically

### Classification Models

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Extra Trees
- AdaBoost
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes

### Regression Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- AdaBoost Regressor
- Support Vector Regressor
- KNN Regressor

---

# 🚀 Hyperparameter Tuning

The project automatically performs hyperparameter optimization using

- GridSearchCV
- RandomizedSearchCV

to improve model performance.

---

# 📊 Performance Metrics

## Classification

- Accuracy
- Precision
- Recall
- F1 Score
- Cross Validation Score

## Regression

- MAE
- MSE
- RMSE
- R² Score
- Cross Validation Score

---

# 📈 Visualizations

The application provides several visualizations including:

- Model Performance Comparison
- Accuracy Comparison
- Precision Comparison
- Recall Comparison
- F1 Score Comparison
- Regression Metrics Comparison
- Feature Importance (Tree-based models)
- Correlation Heatmap
- Dataset Overview

---

# 🏆 Best Model Selection

After training all models, the application automatically

- Compares all models
- Selects the best performing model
- Displays its evaluation metrics
- Saves the trained model

---

# 💾 Model Saving

The project automatically saves

- Best trained model
- Scaler
- Label Encoders
- Target Encoder

using **Joblib**

Example:

```
saved_models/
│
├── best_model.pkl
├── scaler.pkl
├── label_encoders.pkl
└── target_encoder.pkl
```

---

# 📂 Project Structure

```
AutoML-Model-Comparison-System/
│
├── app.py
├── preprocessing.py
├── models.py
├── prediction.py
├── evaluation.py
├── visualization.py
├── train.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── saved_models/
│
└── assets/
```

---

# ⚙️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- SciPy

---

# 🖥️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AutoML-Model-Comparison-System.git
```

Move into the project

```bash
cd AutoML-Model-Comparison-System
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📝 How to Use

### Step 1

Upload a CSV dataset.

---

### Step 2

Select the target column.

---

### Step 3

Choose

- Auto Detect
- Classification
- Regression

---

### Step 4

Click **Train Models**

---

### Step 5

The application will

- Preprocess the data
- Encode categorical columns
- Scale numerical features
- Train all supported ML models
- Perform Hyperparameter Tuning
- Compare results
- Select the best model
- Save the trained model

---

# 📊 Sample Output

The application displays

- Model Comparison Table
- Best Model
- Cross Validation Score
- Hyperparameters
- Feature Importance
- Prediction Results

---

# 📌 Future Improvements

- Deep Learning Models
- XGBoost
- LightGBM
- CatBoost
- SHAP Explainability
- Automated Feature Engineering
- Model Deployment
- Export trained models
- Download prediction reports
- MLflow Integration
- Docker Support

---

# 📷 Screenshots

You can add screenshots here.

Example

```
screenshots/
│
├── homepage.png
├── upload.png
├── results.png
├── feature_importance.png
```

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository

2. Create a new branch

```
git checkout -b feature-name
```

3. Commit your changes

```
git commit -m "Added new feature"
```

4. Push

```
git push origin feature-name
```

5. Open a Pull Request

---

# 👨‍💻 Author

**Mohammed Rishal**

Data Science Trainee

GitHub: https://github.com/MohammedRishal6257

LinkedIn: www.linkedin.com/in/mohammed-rishal-34836131a

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.
