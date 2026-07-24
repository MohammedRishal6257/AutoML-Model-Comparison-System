import time
import traceback
import pandas as pd
import numpy as np
from sklearn.model_selection import ( 
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)

from scipy.stats import randint
from pandas.api.types import (
    is_numeric_dtype,
    is_integer_dtype,
    is_float_dtype,
)




print("Using models.py:", __file__)

# ===========================================
# Classification Models
# ===========================================

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from scipy.sparse import issparse

# ===========================================
# Regression Models
# ===========================================

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# ===========================================
# Metrics
# ===========================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import cross_val_score


# ===================================================
# TRAIN MODELS
# ===================================================

def train_models(
    X_train,
    X_test,
    y_train,
    y_test,
    problem_type_option="Auto Detect"
):

    

    # ------------------------------------------
    # Detect Problem Type
    # ------------------------------------------


    y = pd.Series(y_train)

    if problem_type_option == "Classification":
        problem_type = "classification"

    elif problem_type_option == "Regression":
        problem_type = "regression"

    else:

        if is_numeric_dtype(y):

            if is_float_dtype(y):

                # decimal target
                problem_type = "regression"

            elif is_integer_dtype(y):

                if y.nunique() <= min(20, len(y) * 0.05):
                    problem_type = "classification"
                else:
                    problem_type = "regression"

            else:
                problem_type = "regression"

        else:
            problem_type = "classification"

    print("Detected:", problem_type)

    print("=" * 80)
    print("Problem Type :", problem_type)
    print("Target dtype :", y.dtype)
    print("Unique values:", y.nunique())
    print("=" * 80)

    # ------------------------------------------
    # Classification Models
    # ------------------------------------------

    if problem_type == "classification":

        models = {

            "Logistic Regression":
                LogisticRegression(max_iter=1000),

            "Decision Tree":
                DecisionTreeClassifier(random_state=42),

            "Random Forest":
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                ),

            "Gradient Boosting":
                GradientBoostingClassifier(random_state=42),

            "Extra Trees":
                ExtraTreesClassifier(random_state=42,
                                     n_jobs=-1
                ),

            "AdaBoost":
                AdaBoostClassifier(random_state=42),

            "Support Vector Machine":
                SVC(kernel="rbf", random_state=42),

            "KNN":
                KNeighborsClassifier(n_jobs=-1),

        }


        param_grids = {
        
                        "Logistic Regression": {
                            "C":[0.01,0.1,1,10],
                            "solver":["lbfgs"]
                        },
        
                        "Decision Tree": {
                            "max_depth":[None,5,10,20],
                            "min_samples_split":[2,5,10]
                        },
        
                        "Random Forest":{
        
                            "n_estimators":[100,200,300,500],
        
                            "max_depth":[None,10,20,30],
        
                            "min_samples_split":[2,5,10],
        
                            "min_samples_leaf":[1,2,4],
        
                            "max_features":["sqrt","log2"]
                        },
        
                        "Gradient Boosting": {
                            "n_estimators":[100,200],
                            "learning_rate":[0.01,0.1],
                            "max_depth":[3,5]
                        },
        
                        "Extra Trees": {
                            "n_estimators":[100,200],
                            "max_depth":[None,10,20]
                        },
        
                        "KNN":{
                            "n_neighbors":[3,5,7,9]
                        },
        
                        "Support Vector Machine":{
                            "C":[0.1,1,10],
                            "kernel":["linear","rbf"]
                        }
        
    }

        if issparse(X_train):
            models["Multinomial Naive Bayes"] = MultinomialNB()
        else:
            models["Gaussian Naive Bayes"] = GaussianNB()




    # ------------------------------------------
    # Regression Models
    # ------------------------------------------

    else:

        models = {

            "Linear Regression":
                LinearRegression(),

            "Decision Tree":
                DecisionTreeRegressor(random_state=42),

            "Random Forest":
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                ),

            "Gradient Boosting":
                GradientBoostingRegressor(random_state=42),

            "Extra Trees":
                ExtraTreesRegressor(random_state=42,n_jobs=-1),

            "AdaBoost":
                AdaBoostRegressor(random_state=42),

            "Support Vector Regression":
                SVR(kernel="rbf"),

            "KNN Regressor":
                KNeighborsRegressor(n_jobs=-1),

        }



        param_grids = {

            "Linear Regression": {},

            "Decision Tree":{
                "max_depth":[None,5,10,20]
            },

            "Random Forest":{

                "n_estimators":[100,200,300,500],

                "max_depth":[None,10,20,30],

                "min_samples_split":[2,5,10],

                "min_samples_leaf":[1,2,4]
            },

            "Gradient Boosting":{
                "learning_rate":[0.01,0.1],
                "n_estimators":[100,200]
            },

            "Extra Trees":{
                "n_estimators":[100,200],
                "max_depth":[None,10]
            },

            "KNN Regressor":{
                "n_neighbors":[3,5,7]
            },

            "Support Vector Regression":{
                "C":[0.1,1,10],
                "kernel":["linear","rbf"]
            }

        }

    # ------------------------------------------
    # Training
    # ------------------------------------------
    
    results = []
    
    best_model = None
    best_model_name = None
    best_score = float("-inf")

    # ------------------------------------------

    print("=" * 80)
    print("Problem Type :", problem_type)
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train dtype:", np.array(y_train).dtype)
    print("y_test dtype :", np.array(y_test).dtype)
    print("Unique y values:", np.unique(y_train)[:20])
    print("=" * 80)

    # ------------------------------------------
    
    for name, model in models.items():


        try:

            start = time.time()

            # -----------------------------------------
            # Hyperparameter Tuning
            # -----------------------------------------

            search = None
            best_params = "-"
            

            if name in param_grids and len(param_grids[name]) > 0:

                # Small models -> Grid Search
                if name in [
                    "Logistic Regression",
                    "Decision Tree",
                    "Linear Regression"
                ]:

                    search = GridSearchCV(
                        estimator=model,
                        param_grid=param_grids[name],
                        cv=5,
                        scoring="accuracy" if problem_type=="classification" else "r2",
                        n_jobs=-1
                    )

                else:

                    from math import prod

                    total_possible = prod(
                        len(v) if isinstance(v, list) else 1
                        for v in param_grids[name].values()
                    )

                    search = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=param_grids[name],
                        n_iter=min(10, total_possible),
                        cv=5,
                        scoring="accuracy" if problem_type=="classification" else "r2",
                        random_state=42,
                        n_jobs=-1
                    )

                search.fit(X_train, y_train)

                model = search.best_estimator_
                best_params = search.best_params_

            else:

                model.fit(X_train, y_train)

            training_time = round(time.time() - start,3)

            predictions = model.predict(X_test)

            # ==================================
            # Classification
            # ==================================

            if problem_type == "classification":

                accuracy = accuracy_score(
                    y_test,
                    predictions
                )

                precision = precision_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                recall = recall_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )

                f1 = f1_score(
                    y_test,
                    predictions,
                    average="weighted",
                    zero_division=0
                )


                cv = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="accuracy" if problem_type=="classification" else "r2",
                    n_jobs=-1
                ).mean()

                results.append({

                    "Model": name,

                    "Accuracy": round(accuracy*100,2),

                    "Precision": round(precision*100,2),

                    "Recall": round(recall*100,2),

                    "F1 Score": round(f1*100,2),

                    "CV Score": round(cv*100,2),

                    "Training Time (s)": training_time,

                    "Best Parameters": (
                        str(best_params)
                        if best_params != "-"
                        else "-"
)
                })

                score = accuracy

            # ==================================
            # Regression
            # ==================================

            else:

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                mse = mean_squared_error(
                    y_test,
                    predictions
                )

                rmse = mse ** 0.5

                r2 = r2_score(
                    y_test,
                    predictions
                )
                
                cv = cross_val_score(
                    model,
                    X_train,
                    y_train,
                    cv=5,
                    scoring="accuracy" if problem_type=="classification" else "r2",
                    n_jobs=-1
                ).mean()

                results.append({

                    "Model": name,

                    "MAE": round(mae,2),

                    "MSE": round(mse,2),

                    "RMSE": round(rmse,2),

                    "R2 Score": round(r2,4),

                    "CV Score": round(cv,4),

                    "Training Time (s)": training_time,

                    "Best Parameters": (
                        str(best_params)
                        if best_params != "-"
                        else "-"
                    )
                })

                score = r2

            # ----------------------------------

            if score > best_score:

                best_score = score

                best_model = model

                best_model_name = name

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"❌ {name} failed")
            print("Error:", str(e))
            print("=" * 80)

            import traceback
            traceback.print_exc()

            continue

    # ------------------------------------------

    if len(results)==0:
        raise RuntimeError(
            "No models were trained successfully."
        )
    results_df = pd.DataFrame(results)

    if problem_type == "classification":

        results_df = results_df.sort_values(
            by="Accuracy",
            ascending=False
        )

    else:

        results_df = results_df.sort_values(
            by="R2 Score",
            ascending=False
        )

    results_df.reset_index(
        drop=True,
        inplace=True
    )

    print("Problem Type:", problem_type)
    print(results_df)
    print(best_model)
    print(best_model_name)



    return (
        results_df,
        best_model,
        best_model_name,
        best_score,
        problem_type
    )

    
print("train_models() loaded successfully")
