from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class PredictionInput:
    age: int
    gender: str
    num_adults: int
    num_children: int
    budget: str
    travel_month: int
    preferences: List[str]


@dataclass
class PredictionResult:
    predictions: Dict[str, float]
    top_destinations: List[str]
    top_probabilities: List[float]


class FeatureInfo:
    def __init__(self, data: Dict[str, Any]):
        self.all_features = data.get('all_features', [])
        self.categorical_features = data.get('categorical_features', [])
        self.numeric_features = data.get('numeric_features', [])
        self.categorical_values = data.get('categorical_values', {})
        self.target_classes = data.get('target_classes', [])
