# 🤖 AutoML Prediction & Machine Learning Model Comparison System

An end-to-end **AutoML (Automated Machine Learning)** web application built using **Python, Streamlit, and Scikit-learn** that allows users to upload any CSV dataset, automatically preprocess the data, compare multiple Machine Learning models, evaluate their performance, and make predictions using the best-performing model.

---

# 📌 Table of Contents

- Overview
- Features
- Technologies Used
- Project Structure
- Workflow
- Installation
- Usage
- Machine Learning Models
- Evaluation Metrics
- Data Preprocessing
- Visualization
- Prediction Module
- Future Improvements
- Screenshots
- Author

---

# 📖 Overview

Machine Learning projects usually require several repetitive steps such as

- Data Cleaning
- Feature Engineering
- Handling Missing Values
- Encoding Categorical Variables
- Scaling Features
- Model Selection
- Model Evaluation

This project automates the complete workflow.

The user simply uploads a CSV dataset, selects the target column, and the system automatically builds multiple ML models, compares them, selects the best one, and allows real-time predictions.

---

# ✨ Features

## Dataset Analysis

- Upload any CSV dataset
- Dataset Preview
- Number of Rows
- Number of Columns
- Missing Values
- Duplicate Rows
- Column Information
- Automatic Data Type Detection

---

## Automatic Data Preprocessing

The application automatically performs

- Duplicate Removal
- Missing Value Handling
- Date Feature Extraction
- Text Feature Extraction using TF-IDF
- Label Encoding
- Feature Scaling
- Removal of ID, URL, Email and Address Columns

---

## Automatic Problem Detection

Supports

✅ Classification

Examples

- Loan Prediction
- Job Hiring
- Disease Prediction
- Customer Churn

---

✅ Regression

Examples

- House Price Prediction
- Salary Prediction
- Product Cost Prediction

---

## Multiple Machine Learning Models

The application trains multiple models and compares their performance automatically.

### Classification Models

- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Extra Trees Classifier
- AdaBoost Classifier
- Support Vector Machine
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Multinomial Naive Bayes (Sparse Data)

---

### Regression Models

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor
- AdaBoost Regressor
- Support Vector Regressor
- K-Nearest Neighbor Regressor

---

## Automatic Best Model Selection

After training every model, the application automatically selects the model with the highest performance score.

---

## Model Evaluation

### Classification

Displays

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- ROC Curve (Binary Classification)

---

### Regression

Displays

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## Visualization

Automatic visualizations include

- Missing Value Heatmap
- Correlation Heatmap
- Target Distribution
- Boxplots
- Pairplots
- Distribution Plots
- Feature Importance
- Model Comparison Graph

---

## Prediction

After training

- User enters new feature values
- Data is automatically encoded
- Data is scaled
- Best model predicts output

---

# 🛠 Technologies Used

## Programming Language

- Python 3.12+

---

## Frontend

- Streamlit

---

## Data Processing

- Pandas
- NumPy

---

## Machine Learning

- Scikit-learn

---

## Visualization

- Matplotlib
- Seaborn

---

## Text Processing

- TF-IDF Vectorizer

---

## Utilities

- SciPy

---

# 📂 Project Structure

```
AutoML-Model-Comparison-System/

│

├── app.py

├── preprocessing.py

├── models.py

├── evaluation.py

├── prediction.py

├── visualization.py

├── requirements.txt

├── README.md

├── datasets/

│      sample.csv

│

└── assets/
```

---

# ⚙ Workflow

```
User Uploads CSV
        │
        ▼
Dataset Analysis
        │
        ▼
Automatic Preprocessing
        │
        ▼
Target Selection
        │
        ▼
Problem Type Detection
        │
        ▼
Model Training
        │
        ▼
Model Comparison
        │
        ▼
Best Model Selection
        │
        ▼
Evaluation
        │
        ▼
Prediction
```

---

# 📊 Automatic Preprocessing

## Duplicate Removal

```python
df.drop_duplicates()
```

---

## Missing Value Handling

Numeric Columns

```
Mean Imputation
```

Categorical Columns

```
Most Frequent Value
```

---

## Date Processing

Example

```
2026-07-23

↓

Year = 2026

Month = 7

Day = 23
```

---

## Text Processing

Long text columns are converted into

```
TF-IDF Features
```

---

## Label Encoding

Example

```
Male

Female

↓

0

1
```

---

## Feature Scaling

StandardScaler is used before model training.

---

# 📊 Machine Learning Workflow

```
Dataset

↓

Preprocessing

↓

Train Test Split

↓

Feature Scaling

↓

Model Training

↓

Evaluation

↓

Prediction
```

---

# 📈 Evaluation Metrics

## Classification

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- ROC Curve

---

## Regression

- MAE
- MSE
- RMSE
- R² Score

---

# 📉 Visualization Modules

The application automatically generates

- Dataset Missing Value Graph
- Correlation Heatmap
- Target Distribution
- Pairplot
- Distribution Plot
- Feature Importance Plot
- Model Comparison Plot
- Confusion Matrix
- ROC Curve

---

# 🔮 Prediction Module

After selecting the best model

Users enter

- Numeric Values
- Categorical Values
- Date Values

The application automatically

- Encodes
- Scales
- Predicts

and displays

```
Prediction

Confidence (Classification)

Input Summary
```

---

# 🚀 Installation

Clone Repository

```bash
git clone https://github.com/yourusername/AutoML-Model-Comparison-System.git
```

Move into project

```bash
cd AutoML-Model-Comparison-System
```

Create Virtual Environment

```bash
python -m venv venv
```

Activate

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install Dependencies

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run app.py
```

---

# 📌 Example Datasets

Works with

- Employee Salary Dataset
- House Price Dataset
- Loan Prediction Dataset
- Student Performance Dataset
- Customer Churn Dataset
- Job Hiring Dataset
- Product Dataset
- Sales Dataset

---

# 🌟 Advantages

- Fully Automatic
- Beginner Friendly
- Supports Classification & Regression
- Automatic Preprocessing
- Multiple Model Comparison
- Interactive Dashboard
- Real-Time Prediction
- Feature Importance Visualization

---

# 🚧 Future Improvements

- Hyperparameter Tuning
- Cross Validation Score
- Model Download
- XGBoost
- CatBoost
- LightGBM
- Deep Learning Support
- SHAP Explainability
- LIME Explainability
- PDF Report Generation
- Cloud Deployment
- Database Integration

---

# 📷 Screenshots

Add screenshots here

```
Dataset Preview

Model Leaderboard

Confusion Matrix

Prediction Screen

Feature Importance
```

---

# 👨‍💻 Author

**Mohammed Rishal**

- BCA Graduate
- Diploma in Data Science
- Python Developer
- Machine Learning Enthusiast

---

# 📄 License

This project is developed for educational and research purposes.

---

# ⭐ If you like this project

Please consider giving it a ⭐ on GitHub.