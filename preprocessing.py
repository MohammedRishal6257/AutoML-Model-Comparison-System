import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix, issparse
from pandas.api.types import (
    is_numeric_dtype,
    is_integer_dtype,
    is_float_dtype,
)


# ======================================================
# Detect Column Type
# ======================================================

def detect_column_type(series):

    # Column name
    column_name = str(series.name).lower()

    # Numeric
    if is_numeric_dtype(series):
        return "numeric"

    sample = series.dropna().astype(str)

    if len(sample) == 0:
        return "empty"

    # Address detection
    address_keywords = [
        "address",
        "street",
        "road",
        "lane",
        "avenue",
        "city",
        "state",
        "zipcode",
        "zip",
        "postal"
    ]

    if any(word in column_name for word in address_keywords):
        return "address"

    # URL
    if sample.str.contains("http", case=False).mean() > 0.8:
        return "url"

    # Email
    if sample.str.contains("@").mean() > 0.8:
        return "email"

    # Date
    converted = pd.to_datetime(
        sample,
        errors="coerce",
        format="mixed"
    )

    if converted.notna().mean() > 0.8:
        return "date"

    # Long text (Reviews, Comments, Description etc.)
    if sample.str.len().mean() > 40:
        return "text"

    # ID columns
    if sample.nunique() == len(sample):
        return "id"

    return "categorical"

# ======================================================
# Get Valid Target Columns
# ======================================================


def get_target_columns(df):

    targets = []

    for col in df.columns:

        col_type = detect_column_type(df[col])

        if col_type not in [
            "id",
            "address",
            "url",
            "email",
            "date"
        ]:
            targets.append(col)

    return targets

# ======================================================
# Preprocessing
# ======================================================

def preprocess_data(df, target_column):

    df = df.copy()

    df.drop_duplicates(inplace=True)

    df.replace(
        ["?", "NA", "N/A", "NULL", "null", "-", "", " "],
        np.nan,
        inplace=True
    )

    label_encoders = {}

    text_columns = []

    drop_columns = []

    # --------------------------------------------------

    for col in df.columns:

        if col == target_column:
            continue

        col_type = detect_column_type(df[col])

        print(col, "->", col_type)

        if col_type == "date":

            converted = pd.to_datetime(
                df[col],
                errors="coerce",
                format="mixed"
            )

            df[col + "_Year"] = converted.dt.year
            df[col + "_Month"] = converted.dt.month
            df[col + "_Day"] = converted.dt.day

            drop_columns.append(col)

        elif col_type in ["url", "email", "id", "address"]:

            drop_columns.append(col)

        elif col_type == "text":

            text_columns.append(col)

    df.drop(columns=drop_columns, inplace=True, errors="ignore")

    # --------------------------------------------------

    X = df.drop(columns=[target_column])

    y = df[target_column]

    # --------------------------------------------------
    # Remove constant columns
    # --------------------------------------------------

    constant = [
        c for c in X.columns
        if X[c].nunique(dropna=False) <= 1
    ]

    X.drop(columns=constant, inplace=True)

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    numeric_cols = X.select_dtypes(include="number").columns

    categorical_cols = [
        c for c in X.select_dtypes(exclude="number").columns
        if c not in text_columns
    ]

    if len(numeric_cols):

        X[numeric_cols] = SimpleImputer(
            strategy="mean"
        ).fit_transform(X[numeric_cols])

    if len(categorical_cols):

        X[categorical_cols] = SimpleImputer(
            strategy="most_frequent"
        ).fit_transform(X[categorical_cols])

    # --------------------------------------------------
    # Encode categorical
    # --------------------------------------------------

    for col in categorical_cols:

        le = LabelEncoder()

        X[col] = le.fit_transform(
            X[col].astype(str)
        )

        label_encoders[col] = le

    # --------------------------------------------------
    # TF-IDF Text Columns
    # --------------------------------------------------

    tfidf_features = []

    X = X.drop(columns=text_columns, errors="ignore")

    feature_names = list(X.columns)

    for col in text_columns:

        vectorizer = TfidfVectorizer(max_features=100)

        matrix = vectorizer.fit_transform(df[col].fillna("").astype(str))

        tfidf_features.append(matrix)

        feature_names.extend(
            [f"{col}_{word}" for word in vectorizer.get_feature_names_out()]
        )

    if len(tfidf_features):

        X_processed = hstack([csr_matrix(X.values)] + tfidf_features)

    else:

        X_processed = csr_matrix(X.values)

    # --------------------------------------------------
    # Target
    # --------------------------------------------------
    
    
    
    target_encoder = None

    if not is_numeric_dtype(y):

        if y.mode().empty:
            raise ValueError("Target column contains only missing values.")

        y = y.fillna(y.mode()[0])

        target_encoder = LabelEncoder()

        y = target_encoder.fit_transform(y.astype(str))

    else:

        if y.isna().all():
            raise ValueError("Target column contains only missing values.")

        y = y.fillna(y.mean())


    print("Feature Names:", feature_names)
    print("Number of Features:", len(feature_names))
    print("X Shape:", X.shape)
    print("Target Shape:", y.shape)

    if X.shape[1] == 0:
        raise ValueError(
            "No feature columns remain after preprocessing."
        )

    # --------------------------------------------------
    # Split
    # --------------------------------------------------

    # Only stratify for classification targets
    stratify = None
    if not is_numeric_dtype(y):
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.20,
        random_state=42,
        stratify=stratify
    )

    # --------------------------------------------------
    # Scale
    # --------------------------------------------------

    if issparse(X_train):
        scaler = StandardScaler(with_mean=False)
    else:
        scaler = StandardScaler()

    # Fit only on training data
    X_train = scaler.fit_transform(X_train)

    # Transform test data
    X_test = scaler.transform(X_test)

    # --------------------------------------------------
    # Check NaN
    # --------------------------------------------------

    if issparse(X_train):
        if np.isnan(X_train.data).any():
            raise ValueError("NaN values found after scaling.")
    else:
        if np.isnan(X_train).any():
            raise ValueError("NaN values found after scaling.")

    # --------------------------------------------------
    # Debug
    # --------------------------------------------------

    print("X_train:", X_train.shape)
    print("X_test :", X_test.shape)
    print("y_train:", np.array(y_train).shape)
    print("y_test :", np.array(y_test).shape)

    print("Target dtype :", y.dtype)
    print("Scaler expects:", scaler.n_features_in_)
    print("Feature count :", len(feature_names))

    if scaler.n_features_in_ != len(feature_names):
        feature_names = feature_names[:scaler.n_features_in_]

    return (
        X_train,
        X_test,
        np.array(y_train),
        np.array(y_test),
        scaler,
        label_encoders,
        target_encoder,
        feature_names
    )
