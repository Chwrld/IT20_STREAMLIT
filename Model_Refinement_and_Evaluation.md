# 6. Model Refinement and Performance Improvement

Predictive performance and model quality were significantly improved beyond the initial baseline models through a multi-step refinement process designed to remove bias, prevent data leakage, and extract more granular signals from user behavior.

**Refinement Strategies:**
1. **Data Leakage Resolution:** Initially, destination-level attributes (`State`, `Type`, `BestTimeToVisit`) mapped perfectly to `DestinationName`, causing the models to simply "memorize" destinations rather than learning user preferences. These attributes were entirely excluded, forcing the model to learn authentic supervised mappings from user profile data (e.g., Age, Gender, Number of Adults/Children).
2. **Reliability Filtering:** To capture strong recommendation signals, the training data was filtered to only include trips with high satisfaction (`ExperienceRating >= 4`). Learning only from successful trips aligns the recommender's loss function directly with high user satisfaction.
3. **Feature Engineering:**  
   - Extracted `TravelMonth` from the raw `VisitDate` to help algorithms capture seasonal behavior changes.
   - Expanded the text-based `Preferences` column into one-hot encoded binary columns (`Pref_Relaxation`, `Pref_Adventure`, `Pref_Culture`, `Pref_Spiritual`). This effectively empowered the models to find compound preference combinations.
4. **Hyperparameter Tuning & Class Imbalance:** To prevent models from defaulting to the most popular destinations, a balanced `class_weight` was applied mathematically across the cost function of the algorithms. Exhaustive Grid Search was run on hyperparameters (e.g., for KNN's `n_neighbors`, `weights`, `p`) to find optimal separation boundaries, and early stopping was utilized for Neural Networks (MLP) to prevent overfitting.
5. **Exploring Additional Algorithms:** 14 diverse algorithms ranging from Linear (Logistic Regression, SVM) and Non-Linear (KNN, Naive Bayes), to powerful Ensembles (Random Forest, AdaBoost, Bagging) and Neural Networks were evaluated. An advanced Meta-Learner ('Stacking Ensemble') was introduced combining multiple base estimators.

**Justification & Business Value:**
Without these refinements, initial predictors might achieve artificially high accuracy strictly through data leakage or popularity bias, completely failing on unseen user profiles. By ensuring rigorous feature engineering and addressing class imbalance, we achieved predictive accuracy that generalizes. High-accuracy, personalized destination recommendations convert browsers into active travelers more effectively, driving higher engagement and tangible revenue growth for the travel platform logic.

---

# 7. Evaluation Metrics and Results

Because accuracy alone is flawed when assessing multi-class recommendation models, a broader tiered evaluation architecture was designed to measure model performance accurately.

**Evaluation Metrics Used:**
- **F1-Score (Weighted):** An ideal metric for our multi-class task, balancing precision and recall. A weighted approach ensures that minority preference profiles (less popular destinations) are proportionally represented, penalizing algorithms that only successfully predict the majority classes.
- **Top-3 Accuracy:** Reflects the probability that the user’s true preferred destination is among the model’s top 3 ranked recommendations. Since the interactive app interface displays exactly 3 recommendations, this metric perfectly mirrors the end-user product experience and business goal.
- **5-Fold Cross-Validation F1-Score (CV-F1):** Used to prove generalization capabilities across 5 distinct train/test splits, verifying that results are not due to lucky random sampling.
- **Test Accuracy:** Basic correctness check across the held-out test subset.

**Presenting & Interpreting the Results:**
Instead of picking the model with the highest single score (which often hides flaws in other areas), a **Harmonic Mean (Consistency Score)** was engineered. This required normalizing F1-Score, Test Accuracy, and Top-3 Accuracy into identical scales, then taking their harmonic mean.
- A model scoring high on accuracy but low on Top-3 ranking is heavily penalized by the harmonic mean. 
- The resulting algorithm comparison table tiers algorithms structurally by: `Top-3 Accuracy` → `F1-Score` → `Cross-Validation F1`. 
- By enforcing mathematical model agreement checks, it was proven that different models made varied predictions indicating true semantic learning rather than identical biases. Finally, the model boasting the highest Harmonic Mean and verified cross-validation scores was exported (`best_model.joblib`), ensuring deployment of an algorithm making highly consistent, high-confidence recommendations suitable for real-world decision-making.
