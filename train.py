import os
import joblib
import pandas as pd

from preprocessing import preprocess_data
from models import train_models


# =====================================
# SETTINGS
# =====================================

DATASET_PATH = "datasets/sample.csv"

TARGET_COLUMN = "Target"

SAVE_FOLDER = "saved_models"

# =====================================
# Check Dataset
# =====================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

# =====================================
# Load Dataset
# =====================================

print("Loading Dataset...")

df = pd.read_csv(DATASET_PATH)

print(df.head())

print(f"\nRows : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")

# =====================================
# Preprocess
# =====================================

print("\nPreprocessing Dataset...")

(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    label_encoders,
    target_encoder,
    feature_columns
) = preprocess_data(
    df,
    TARGET_COLUMN
)

# =====================================
# Train Models
# =====================================

print("\nTraining Models...")

results_df, best_model, problem_type = train_models(

    X_train,
    X_test,
    y_train,
    y_test

)

# =====================================
# Results
# =====================================

print("\n============================")
print("MODEL COMPARISON")
print("============================\n")

print(results_df)

print("\nBest Model")

print(results_df.iloc[0]["Model"])

print(f"\nProblem Type : {problem_type}")

# =====================================
# Create Folder
# =====================================

os.makedirs(
    SAVE_FOLDER,
    exist_ok=True
)

# =====================================
# Save Files
# =====================================

joblib.dump(
    best_model,
    os.path.join(
        SAVE_FOLDER,
        "best_model.pkl"
    )
)

joblib.dump(
    scaler,
    os.path.join(
        SAVE_FOLDER,
        "scaler.pkl"
    )
)

joblib.dump(
    label_encoders,
    os.path.join(
        SAVE_FOLDER,
        "label_encoders.pkl"
    )
)

joblib.dump(
    target_encoder,
    os.path.join(
        SAVE_FOLDER,
        "target_encoder.pkl"
    )
)

joblib.dump(
    feature_columns,
    os.path.join(
        SAVE_FOLDER,
        "feature_columns.pkl"
    )
)

# =====================================
# Finished
# =====================================

print("\n============================")
print("Training Completed Successfully")
print("============================")

print(f"\nFiles saved inside '{SAVE_FOLDER}'")

print("""
Saved Files

best_model.pkl
scaler.pkl
label_encoders.pkl
target_encoder.pkl
feature_columns.pkl
""")
