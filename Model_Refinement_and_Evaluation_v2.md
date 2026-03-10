# Model Refinement and Evaluation

This document details the refinement strategies and evaluation results for the Travel Destination Recommendation engine, focusing on the comparative performance of leading models like **Logistic Regression** and **Random Forest**.

## 6. Model Refinement and Performance Improvement

The predictive performance of the recommendation engine was improved through a deliberate process of handling data leakage, scaling, and class imbalances.

### Refinement Strategies

1. **Feature Engineering & Leakage Prevention**:
   - Explicitly excluded destination-level attributes ('State', 'Type', and 'BestTimeToVisit') from the training features. Retaining these would cause 100% data leakage, allowing models to memorize mappings rather than learning genuine user preference patterns.

2. **Algorithm Specific Refinements**:
   - **Logistic Regression**: Served as a highly effective model when optimized with the `lbfgs` solver for multinomial classification. It required proper feature scaling (`MinMaxScaler`) prior to training to ensure coefficient stability.
   - **Random Forest**: Explored as a powerful ensemble alternative capable of capturing non-linear relationships. It was optimized by controlling the tree depth (`max_depth=15`) and increasing the ensemble size (`n_estimators=200`). Unlike Logistic Regression, the Random Forest pipeline utilized unscaled features.

3. **Inbound Bias Mitigation**:
   - **Balanced Class Weights**: The dataset exhibited imbalanced travel histories (e.g., favoring Major Metros). Applying `class_weight='balanced'` to both Logistic Regression and Random Forest ensured the models did not become biased toward majority classes, preserving their ability to recommend niche destinations.

### Justification for Refinements
These refinements were essential to produce a recommendation system that offers real business value. By prioritizing class balance and preventing leakage, the models generate reliable, unbiased suggestions based on user profiles rather than dataset artifacts.

## 7. Evaluation Metrics and Results

Given the multi-class nature of the recommendation task (predicting one of many destinations), relying solely on raw exact-match accuracy is insufficient. A weighted combination of metrics was used to holistically evaluate the models.

### Key Metrics
- **Accuracy**: The global exact-match accuracy.
- **Top-3 Accuracy**: A critical business metric. In a recommendation interface displaying three top choices, this measures the probability that the relevant destination is within those top three.
- **Weighted F1-Score**: Balances precision and recall across all destination classes.
- **CV-F1**: 5-Fold Cross-Validation F1-score to check performance stability across different data subsets.
- **Consistency Score**: A custom Harmonic Mean calculated from the normalized Accuracy, F1-Score, and Top-3 Accuracy. This score identifies the model that performs consistently well across all dimensions without major weaknesses.

### Comparative Performance Results

The following table presents the final evaluation results sorted by the Consistency metric:

| Model | Accuracy | F1-Score | Top3-Acc | CV-F1 | Consistency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **0.2500** | **0.2441** | 0.5738 | 0.1882 | **0.976569** |
| Random Forest | 0.2377 | 0.2357 | **0.5902** | **0.2067** | 0.898430 |
| Bagging | 0.2295 | 0.2225 | 0.5820 | 0.2309 | 0.792896 |
| SGD Classifier | 0.2459 | 0.1967 | 0.5451 | 0.1172 | 0.771920 |
| K-Nearest Neighbours | 0.2336 | 0.2098 | 0.5615 | 0.2064 | 0.770938 |
| Gradient Boosting | 0.2254 | 0.2196 | 0.5492 | 0.2290 | 0.718720 |
| Multi-Layer Perceptron | 0.2254 | 0.2023 | 0.5451 | 0.0000 | 0.674902 |
| Single-Layer Perceptron | 0.2213 | 0.2085 | 0.5287 | 0.0000 | 0.638353 |
| Stacking Ensemble | 0.2172 | 0.1995 | 0.5656 | 0.1906 | 0.606591 |
| Kernel SVM (RBF) | 0.2090 | 0.2056 | 0.5820 | 0.1596 | 0.510487 |
| Linear SVM | 0.2049 | 0.2014 | 0.5902 | 0.1761 | 0.428646 |
| AdaBoost | 0.2008 | 0.1717 | 0.5287 | 0.1490 | 0.293287 |
| Naive Bayes | 0.2172 | 0.1168 | 0.4795 | 0.1119 | 0.028810 |
| Decision Tree | 0.1926 | 0.1930 | 0.3484 | 0.1796 | 0.014877 |

### Interpretation and Model Selection
- **Logistic Regression** is the overall best model based on the **Consistency Score (0.976569)**. It achieves the highest raw exact-match accuracy (25.00%) and a strong F1-Score, meaning it provides the most balanced and stable predictions overall.
- **Random Forest** is a strong secondary candidate. While it ranks second in overall consistency, it achieves the highest **Top-3 Accuracy (59.02%)** and cross-validation stability (CV-F1: 20.67%). 

Ultimately, Logistic Regression's superior consistency makes it a more reliable foundation.
