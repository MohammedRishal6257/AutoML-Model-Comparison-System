import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

plt.style.use("ggplot")

# ======================================================
# Model Comparison
# ======================================================

def plot_results(results_df):

    st.subheader("📊 Model Comparison")

    fig, ax = plt.subplots(figsize=(10,5))

    if "Accuracy" in results_df.columns:

        sns.barplot(
            data=results_df,
            x="Accuracy",
            y="Model",
            hue="Model",
            palette="viridis",
            legend=False,
            ax=ax
        )

        ax.set_xlabel("Accuracy (%)")

    elif "R2 Score" in results_df.columns:

        sns.barplot(
            data=results_df,
            x="R2 Score",
            y="Model",
            hue="Model",
            palette="viridis",
            legend=False,
            ax=ax
        )

        ax.set_xlabel("R² Score")

    st.pyplot(fig)

# ======================================================
# Missing Values
# ======================================================

def plot_missing_values(df):

    st.subheader("📉 Missing Values")

    missing = df.isnull().sum()

    fig, ax = plt.subplots(figsize=(10,4))

    sns.barplot(
        x=missing.index,
        y=missing.values,
        hue=missing.index,
        palette="rocket",
        legend=False,
        ax=ax
    )

    plt.xticks(rotation=45)

    ax.set_ylabel("Missing Values")

    st.pyplot(fig)


# ======================================================
# Correlation Heatmap
# ======================================================

def plot_heatmap(df):

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return

    st.subheader("🔥 Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(10,8))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        linewidths=0.5,
        ax=ax
    )

    st.pyplot(fig)


# ======================================================
# Target Distribution
# ======================================================

def plot_target_distribution(df, target):

    st.subheader("🎯 Target Distribution")

    fig, ax = plt.subplots(figsize=(8,5))

    if df[target].dtype == "object" or df[target].nunique() <= 20:

        sns.countplot(
            x=df[target],
            hue=df[target],
            palette="Set2",
            legend=False,
            ax=ax
        )

        plt.xticks(rotation=45)

    else:

        sns.histplot(
            df[target],
            bins=30,
            kde=True,
            color="steelblue",
            ax=ax
        )

    st.pyplot(fig)


# ======================================================
# Feature Importance
# ======================================================

def plot_feature_importance(model, feature_names):

    if not hasattr(model, "feature_importances_"):
        return

    st.subheader("⭐ Feature Importance")

    importances = model.feature_importances_

    # Make both lists the same length
    length = min(len(feature_names), len(importances))

    importance = pd.DataFrame({
        "Feature": feature_names[:length],
        "Importance": importances[:length]
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    fig, ax = plt.subplots(figsize=(10,6))

    sns.barplot(

        data=importance,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=ax

    )

    st.pyplot(fig)


# ======================================================
# Box Plot
# ======================================================

def plot_boxplots(df):

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return

    st.subheader("📦 Box Plot")

    fig, ax = plt.subplots(figsize=(12,5))

    sns.boxplot(data=numeric, ax=ax)

    plt.xticks(rotation=90)

    st.pyplot(fig)


# ======================================================
# Pair Plot
# ======================================================

def plot_pairplot(df):

    numeric = df.select_dtypes(include="number")

    if numeric.shape[1] < 2:
        return

    if numeric.shape[1] > 5:
        numeric = numeric.iloc[:, :5]

    st.subheader("🔗 Pair Plot")

    pair = sns.pairplot(numeric)

    st.pyplot(pair.figure)


# ======================================================
# Distribution Plots
# ======================================================

def plot_distributions(df):

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        return

    st.subheader("📈 Feature Distributions")

    for column in numeric.columns:

        fig, ax = plt.subplots(figsize=(7,3))

        sns.histplot(
            numeric[column],
            kde=True,
            ax=ax
        )

        ax.set_title(column)

        st.pyplot(fig)


