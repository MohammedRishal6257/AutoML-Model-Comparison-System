import time
import streamlit as st
import pandas as pd
import joblib
import os


from preprocessing import get_target_columns, preprocess_data,detect_column_type
from models import train_models
from visualization import (
    plot_results,
    plot_heatmap,
    plot_missing_values,
    plot_target_distribution,
    plot_feature_importance,
    plot_boxplots,
    plot_pairplot,
    plot_distributions
)
from evaluation import (
    evaluate_model,
    show_confusion_matrix,
    show_classification_report,
    show_roc_curve,
    show_actual_vs_predicted,
)
from prediction import predict_new_data


# ====================================================
# PAGE CONFIG
# ====================================================

st.set_page_config(
    page_title="AutoML Prediction System",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AutoML Prediction & Machine Learning Model Comparison System")

st.write(
    "Upload any CSV dataset, automatically preprocess it, compare multiple machine learning models, and predict new values."
)

# ====================================================
# SESSION STATE
# ====================================================

if "trained" not in st.session_state:
    st.session_state.trained = False

# ====================================================
# SIDEBAR
# ====================================================

st.sidebar.title("📂 AutoML")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# ====================================================
# LOAD DATASET
# ====================================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.header("📄 Dataset Preview")

    st.dataframe(df)

    # ------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))
    col4.metric("Duplicate Rows", int(df.duplicated().sum()))

    # ------------------------------------

    st.subheader("Column Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Detected Type": [detect_column_type(df[c]) for c in df.columns],
        "Datatype": df.dtypes.astype(str),
        "Missing": df.isnull().sum().values,
        "Unique": df.nunique().values
    })

    st.dataframe(info)

    # ====================================================
    # VISUALIZATION
    # ====================================================

    plot_missing_values(df)

    plot_heatmap(df)

    plot_boxplots(df)

    plot_pairplot(df)

    plot_distributions(df)

    # ====================================================
    # TARGET
    # ====================================================


    target = st.selectbox(
        "🎯 Select Target Column",
        get_target_columns(df)
    )

    plot_target_distribution(df, target)

    problem_type_option = st.radio(
        "🧠 Problem Type",
        [
            "Auto Detect",
            "Classification",
            "Regression"
        ],
        horizontal=True
    )
    
    # ====================================================
    # TRAIN
    # ====================================================

    train = st.button("🚀 Train Models")

    if train:

        progress = st.progress(0)

        start_time = time.time()

        with st.spinner("Preprocessing dataset..."):

            (
                X_train,
                X_test,
                y_train,
                y_test,
                scaler,
                label_encoders,
                target_encoder,
                feature_columns
            ) = preprocess_data(df, target)

        progress.progress(30)

        print("Selected Problem Type:", problem_type_option)

        with st.spinner("Training machine learning models..."):

                results_df, best_model, best_model_name, best_score, problem_type = train_models(
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                    problem_type_option
                )

                os.makedirs("saved_models", exist_ok=True)

                joblib.dump(best_model, "saved_models/best_model.pkl")
                joblib.dump(scaler, "saved_models/scaler.pkl")
                joblib.dump(label_encoders, "saved_models/label_encoders.pkl")
                joblib.dump(target_encoder, "saved_models/target_encoder.pkl")

                model = joblib.load("saved_models/best_model.pkl")

        progress.progress(80)

        metrics = evaluate_model(
            best_model,
            X_test,
            y_test,
            problem_type
        )

        end_time = time.time()

        progress.progress(100)

        st.session_state.update({
            "trained": True,
            "best_model": best_model,
            "best_model_name": best_model_name,
            "best_score": best_score,
            "problem_type": problem_type,
            "metrics": metrics,
            "results": results_df,
            "scaler": scaler,
            "label_encoders": label_encoders,
            "target_encoder": target_encoder,
            "feature_columns": feature_columns,
            "X_test": X_test,
            "y_test": y_test
        })

        st.success("✅ Training Completed")
        st.success(
            f"🏆 Best Model : {st.session_state.best_model_name}"
        )

        st.metric(
            "Training Time",
            f"{end_time-start_time:.2f} sec"
        )

    # ====================================================
    # SHOW RESULTS
    # ====================================================

    if st.session_state.trained:
        results_df = st.session_state.results
        best_model = st.session_state.best_model
        metrics = st.session_state.metrics
        problem_type = st.session_state.problem_type

        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        st.header("🏆 Model Leaderboard")

        st.dataframe(
            results_df.style.highlight_max(
                subset=["Accuracy"] if problem_type=="classification" else ["R2 Score"],
                color="lightgreen"
            ),
            width="stretch"
        )

        print(results_df.columns)
        print(results_df.head())

        plot_results(results_df)

        st.success(
            f"Best Model : {results_df.iloc[0]['Model']}"
        )

        # ========================================

        st.header("📊 Evaluation")

        if problem_type == "classification":

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Accuracy",
                f"{metrics['Accuracy']*100:.2f}%"
            )

            c2.metric(
                "Precision",
                f"{metrics['Precision']*100:.2f}%"
            )

            c3.metric(
                "Recall",
                f"{metrics['Recall']*100:.2f}%"
            )

            c4.metric(
                "F1 Score",
                f"{metrics['F1 Score']*100:.2f}%"
            )

            show_confusion_matrix(
                best_model,
                X_test,
                y_test,
                problem_type
            )

            show_classification_report(
                best_model,
                X_test,
                y_test,
                problem_type
            )

            show_roc_curve(
                best_model,
                X_test,
                y_test,
                problem_type
            )

        else:

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Mean Absolute Error", f"{metrics['MAE']:.2f}")
            c2.metric("Mean Squared Error", f"{metrics['MSE']:.2f}")
            c3.metric("Root Mean Squared Error", f"{metrics['RMSE']:.2f}")
            c4.metric("R² Score", f"{metrics['R2 Score']:.4f}")

            show_actual_vs_predicted(
                best_model,
                X_test,
                y_test,
                problem_type
            )

        plot_feature_importance(
            best_model,
            st.session_state.feature_columns
        )

        # ========================================
        # PREDICTION
        # ========================================

        predict_new_data(
            best_model,
            df,
            target,
            st.session_state.scaler,
            st.session_state.label_encoders,
            st.session_state.target_encoder,
            st.session_state.feature_columns,
            st.session_state.problem_type
        )




#    streamlit run app.py

#    Local URL: http://localhost:8501

#    Network URL: http://192.168.1.45:8501