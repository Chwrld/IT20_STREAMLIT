import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "ml-training" / "models" / "optimized_random_forest_travel.pkl"
PREPROCESSOR_PATH = ROOT / "ml-training" / "models" / "preprocessor.joblib"
LABEL_ENCODER_PATH = ROOT / "ml-training" / "models" / "label_encoder.joblib"
FEATURE_INFO_PATH = ROOT / "ml-training" / "models" / "feature_info_travel.json"
RESULTS_PATH = ROOT / "ml-training" / "models" / "results_dict.joblib"

_model = None
_preprocessor = None
_le = None
_feature_info = None
_results_dict = None


def load_artifacts():
    global _model, _preprocessor, _le, _feature_info, _results_dict

    if _model is None:
        _model = joblib.load(MODEL_PATH)

    if _preprocessor is None:
        _preprocessor = joblib.load(PREPROCESSOR_PATH)

    if _le is None:
        _le = joblib.load(LABEL_ENCODER_PATH)

    if _feature_info is None:
        with open(FEATURE_INFO_PATH, "r") as f:
            _feature_info = json.load(f)

    if _results_dict is None:
        _results_dict = joblib.load(RESULTS_PATH)


def get_feature_info():
    load_artifacts()
    return _feature_info


def get_accuracy():
    global _results_dict
    load_artifacts()
    if _results_dict is None:
        return 76.2
    return _results_dict.get('Optimized Random Forest', {}).get('Accuracy', 0.762) * 100


def predict_from_dict(input_dict: dict) -> dict:
    global _model, _le, _feature_info
    
    # Ensure all artifacts are loaded
    load_artifacts()
    
    # Verify artifacts are loaded
    if _model is None:
        raise RuntimeError("Model failed to load from " + str(MODEL_PATH))
    if _le is None:
        raise RuntimeError("Label encoder failed to load from " + str(LABEL_ENCODER_PATH))
    if _feature_info is None:
        raise RuntimeError("Feature info failed to load from " + str(FEATURE_INFO_PATH))
    
    # Create DataFrame from input
    input_df = pd.DataFrame([input_dict])
    
    # Preprocess to add derived features
    input_df = preprocess_dataframe(input_df)
    
    # Select features
    numeric_features = _feature_info['numeric_features']
    categorical_features = _feature_info['categorical_features']
    input_features = input_df[numeric_features + categorical_features]
    
    # Predict probabilities
    probs = _model.predict_proba(input_features)[0]
    return dict(zip(_le.classes_, probs))


def get_model():
    load_artifacts()
    return _model


def get_label_encoder():
    load_artifacts()
    return _le


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a DataFrame to add derived features as done in the notebook."""
    df = df.copy()
    
    # Compute derived features
    def get_season(month):
        if 3 <= month <= 5:
            return 'Spring'
        elif 6 <= month <= 8:
            return 'Summer'
        elif 9 <= month <= 11:
            return 'Fall'
        else:
            return 'Winter'
    
    df['TravelSeason'] = df['TravelMonth'].apply(get_season)
    
    def get_age_group(age):
        if age <= 25:
            return 'Young'
        elif age <= 35:
            return 'Adult'
        elif age <= 45:
            return 'Middle'
        elif age <= 55:
            return 'Senior'
        else:
            return 'Elder'
    
    df['Age_Group'] = df['Age'].apply(get_age_group)
    
    df['Family_Size'] = df['NumberOfAdults'] + df['NumberOfChildren']
    df['Has_Children'] = (df['NumberOfChildren'] > 0).astype(int)
    
    budget_map = {'Low': 1, 'Medium': 2, 'High': 3}
    df['Budget_Score'] = df['Budget'].map(budget_map)
    
    # Ensure preference columns exist
    if 'Pref_Relaxation' not in df.columns:
        df['Pref_Relaxation'] = 0
    if 'Pref_Adventure' not in df.columns:
        df['Pref_Adventure'] = 0
    if 'Pref_Culture' not in df.columns:
        df['Pref_Culture'] = 0
    if 'Pref_Spiritual' not in df.columns:
        df['Pref_Spiritual'] = 0
    
    df['Preference_Match_Score'] = 0  # Simplified
    df['Avg_Dest_Rating'] = 4.5  # Default
    df['Review_Count'] = 100  # Default
    df['Seasonal_Match'] = 0  # Simplified
    
    return df