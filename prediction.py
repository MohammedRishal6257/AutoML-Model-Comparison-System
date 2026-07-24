import streamlit as st
import pandas as pd
from pandas.api.types import is_numeric_dtype


def predict_new_data(
    best_model,
    df,
    target_column,
    scaler,
    label_encoders,
    target_encoder,
    feature_columns,
    problem_type
):  

    st.markdown("---")
    st.header("🔮 Predict New Data")

    # -------------------------------
    # Check feature columns
    # -------------------------------

    if not feature_columns:
        st.error("No feature columns found.")
        return

    user_input = {}

    # -------------------------------
    # Create input widgets
    # -------------------------------

    # Create one date picker
    selected_date = st.date_input("📅 Select Date")

    for feature in feature_columns:

        if feature == target_column:
            continue

        if feature.endswith("_Year"):
            user_input[feature] = selected_date.year

        elif feature.endswith("_Month"):
            user_input[feature] = selected_date.month

        elif feature.endswith("_Day"):
            user_input[feature] = selected_date.day

        elif feature in df.columns:

            if is_numeric_dtype(df[feature]):

                user_input[feature] = st.number_input(
                    feature,
                    value=float(df[feature].mean())
                )

            else:

                options = sorted(df[feature].dropna().astype(str).unique())

                user_input[feature] = st.selectbox(
                    feature,
                    options
                )




    # -------------------------------
    # Predict
    # -------------------------------

    if st.button("🚀 Predict"):

        input_df = pd.DataFrame([user_input])

        # Encode categorical columns
        for col, encoder in label_encoders.items():

            if col in input_df.columns:

                value = str(input_df.loc[0, col])

                if value in encoder.classes_:

                    input_df[col] = encoder.transform([value])

                else:

                    input_df[col] = 0

        # Match training columns
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        # Ensure the number of features matches what the scaler expects
        input_df = input_df.iloc[:, :scaler.n_features_in_]

        # Debug
        st.write("Feature Columns:", feature_columns)
        st.write("Input Data")
        st.dataframe(input_df)

        # Prevent scaler crash
        if input_df.empty or input_df.shape[1] == 0:
            st.error("No feature columns available for prediction.")
            return

        st.write("Training features:", len(feature_columns))
        st.write("Prediction features:", len(input_df.columns))

        extra = set(input_df.columns) - set(feature_columns)
        missing = set(feature_columns) - set(input_df.columns)

        st.write("Extra columns:", extra)
        st.write("Missing columns:", missing)

        # Scale
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df

        # Predict
        prediction = best_model.predict(input_scaled)

        if target_encoder is not None:
            prediction = target_encoder.inverse_transform(prediction.astype(int))

        # ==============================
        # Display Prediction
        # ==============================

        st.subheader("🎯 Prediction Result")

        pred = prediction[0]

        if problem_type == "classification":

            st.metric(
                "Prediction",
                value=str(pred)
            )

        else:

            st.metric(
                "Prediction",
                value=f"{float(pred):,.2f}"
            )

        if hasattr(best_model, "predict_proba"):

            probs = best_model.predict_proba(input_scaled)

            confidence = probs.max() * 100

            st.info(f"Confidence : {confidence:.2f}%")

            st.progress(float(confidence) / 100)

        st.subheader("📋 Input Summary")

        st.dataframe(pd.DataFrame([user_input]), use_container_width=True)
