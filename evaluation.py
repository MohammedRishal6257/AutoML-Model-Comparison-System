import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==================================================
# Evaluate Model
# ==================================================

def evaluate_model(model, X_test, y_test, problem_type):

    predictions = model.predict(X_test)

    if problem_type == "classification":

        return {

            "Accuracy":
                accuracy_score(y_test, predictions),

            "Precision":
                precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ),

            "Recall":
                recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                ),

            "F1 Score":
                f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

        }

    else:

        mae = mean_absolute_error(y_test, predictions)

        mse = mean_squared_error(y_test, predictions)

        rmse = np.sqrt(mse)

        r2 = r2_score(y_test, predictions)

        return {

            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2 Score": r2

        }


# ==================================================
# Confusion Matrix
# ==================================================

def show_confusion_matrix(model, X_test, y_test, problem_type):

    if problem_type != "classification":
        return

    predictions = model.predict(X_test)


    fig, ax = plt.subplots(figsize=(6,5))

    st.subheader("📊 Confusion Matrix")

    

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        cmap="Blues",
        ax=ax
    )


    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    st.pyplot(fig)


# ==================================================
# Classification Report
# ==================================================

def show_classification_report(model, X_test, y_test, problem_type):

    if problem_type != "classification":
        return

    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0
    )

    st.subheader("📄 Classification Report")

    st.text(report)


# ==================================================
# ROC Curve
# ==================================================

def show_roc_curve(model, X_test, y_test, problem_type):

    if problem_type != "classification":
        return

    classes = np.unique(y_test)

    if len(classes) != 2:
        st.info("ROC Curve is available only for binary classification.")
        return

    if not hasattr(model, "predict_proba"):
        return

    probabilities = model.predict_proba(X_test)[:,1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    st.subheader("📈 ROC Curve")

    fig, ax = plt.subplots(figsize=(6,5))

    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")

    ax.plot([0,1],[0,1],"--")

    ax.legend()

    ax.set_xlabel("False Positive Rate")

    ax.set_ylabel("True Positive Rate")

    st.pyplot(fig)


# ==================================================
# Precision Recall Curve
# ==================================================

def show_precision_recall_curve(model, X_test, y_test, problem_type):

    if problem_type != "classification":
        return

    classes = np.unique(y_test)

    if len(classes) != 2:
        st.info("Precision-Recall Curve is available only for binary classification.")
        return

    if not hasattr(model, "predict_proba"):
        return

    probabilities = model.predict_proba(X_test)[:,1]

    precision, recall, _ = precision_recall_curve(
        y_test,
        probabilities
    )

    st.subheader("📉 Precision-Recall Curve")

    fig, ax = plt.subplots(figsize=(6,5))

    ax.plot(recall, precision)

    ax.set_xlabel("Recall")

    ax.set_ylabel("Precision")

    st.pyplot(fig)


# ==================================================
# Actual vs Predicted
# ==================================================

def show_actual_vs_predicted(model, X_test, y_test, problem_type):

    if problem_type != "regression":
        return

    predictions = model.predict(X_test)

    st.subheader("📈 Actual vs Predicted")

    fig, ax = plt.subplots(figsize=(7,6))

    ax.scatter(
        y_test,
        predictions,
        alpha=0.7
    )

    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color="red"
    )

    ax.set_xlabel("Actual")

    ax.set_ylabel("Predicted")

    st.pyplot(fig)


# ==================================================
# Residual Plot
# ==================================================

def show_residual_plot(model, X_test, y_test, problem_type):

    if problem_type != "regression":
        return

    predictions = model.predict(X_test)

    residuals = y_test - predictions

    st.subheader("📉 Residual Plot")

    fig, ax = plt.subplots(figsize=(7,5))

    ax.scatter(
        predictions,
        residuals,
        alpha=0.7
    )

    ax.axhline(0,color="red")

    ax.set_xlabel("Predicted")

    ax.set_ylabel("Residual")

    st.pyplot(fig)


# ==================================================
# Error Distribution
# ==================================================

def show_error_distribution(model, X_test, y_test, problem_type):

    if problem_type != "regression":
        return

    predictions = model.predict(X_test)

    errors = y_test - predictions

    st.subheader("📊 Error Distribution")

    fig, ax = plt.subplots(figsize=(7,5))

    sns.histplot(
        errors,
        bins=30,
        kde=True,
        color="royalblue",
        ax=ax
    )

    ax.set_xlabel("Prediction Error")

    st.pyplot(fig)
    