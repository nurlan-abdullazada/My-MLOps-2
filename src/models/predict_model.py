import joblib
import pandas as pd


def predict_multisim(data_path, model_path):
    """
    Makes predictions on new data using a trained model.

    Args:
        data_path (str): The path to the new data (in parquet format).
        model_path (str): The path to the trained model.

    Returns:
        pandas.DataFrame: A DataFrame with the original data and predictions.
    """
    # Load the new data
    new_data = pd.read_parquet(data_path)

    # Load the trained model
    model = joblib.load(model_path)

    # Make predictions
    predictions = model.predict(new_data)
    prediction_probabilities = model.predict_proba(new_data)[:, 1]

    # Add predictions to the DataFrame
    new_data["prediction"] = predictions
    new_data["prediction_probability"] = prediction_probabilities

    return new_data


if __name__ == "__main__":
    # Define file paths
    NEW_DATA_PATH = "new_multisim_data.parquet"  # You'll need a new data file
    MODEL_PATH = "multisim_model.joblib"

    # Create a dummy new data file for demonstration
    # In a real scenario, you would have a separate file with new data
    dummy_data = {
        "age": [30, 45, 22],
        "tenure": [365, 1095, 180],
        "age_dev": [1, 2, 0],
        "trf": ["A", "B", "C"],
        "gndr": ["M", "F", "M"],
        "dev_man": ["Samsung", "Apple", "Samsung"],
        "is_dualsim": [1, 0, 1],
        "region": ["Baku", "Ganja", "Baku"],
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df.to_parquet(NEW_DATA_PATH)

    # Make predictions
    predictions_df = predict_multisim(NEW_DATA_PATH, MODEL_PATH)

    # Display the results
    print("Predictions on new data:")
    print(predictions_df)
