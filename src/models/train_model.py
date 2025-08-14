import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def train_model(data_path, model_output_path):
    """
    Trains a model to predict multisimmers.

    Args:
        data_path (str): The path to the training data parquet file.
        model_output_path (str): The path to save the trained model.
    """
    # Load the dataset
    df = pd.read_parquet(data_path)

    # Define features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify numerical and categorical features
    numerical_features = ["age", "tenure", "age_dev"]
    categorical_features = ["trf", "gndr", "dev_man", "is_dualsim", "region"]

    # Create preprocessing pipelines for numerical and categorical data
    numerical_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Create a preprocessor object using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    # Best hyperparameters from Optuna study
    best_params = {
        "n_estimators": 479,
        "max_depth": 10,
        "learning_rate": 0.23936368054948778,
        "subsample": 0.8030896642185361,
        "colsample_bytree": 0.6029059677831862,
        "gamma": 4.994099512054965,
        "lambda": 2.9186755025238025,
        "alpha": 0.5152080664184715,
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }

    # Create the final pipeline with the best parameters
    final_xgb_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", xgb.XGBClassifier(**best_params))]
    )

    # Train the model
    final_xgb_pipeline.fit(X_train, y_train)

    # Evaluate the model
    y_pred_final = final_xgb_pipeline.predict(X_test)
    y_pred_proba_final = final_xgb_pipeline.predict_proba(X_test)[:, 1]

    print("Final Tuned XGBoost Model Performance")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_final):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_final):.4f}")
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred_final))

    # Save the trained model
    joblib.dump(final_xgb_pipeline, model_output_path)
    print(f"Model saved to {model_output_path}")


if __name__ == "__main__":
    # Define file paths
    DATA_PATH = "multisim_dataset.parquet"
    MODEL_OUTPUT_PATH = "multisim_model.joblib"

    # Run the training process
    train_model(DATA_PATH, MODEL_OUTPUT_PATH)
