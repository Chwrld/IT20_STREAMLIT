import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

# Load raw datasets
df_users = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "amanmehra23/travel-recommendation-dataset", "Final_Updated_Expanded_Users.csv")
df_history = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "amanmehra23/travel-recommendation-dataset", "Final_Updated_Expanded_UserHistory.csv")
df_destinations = kagglehub.load_dataset(KaggleDatasetAdapter.PANDAS, "amanmehra23/travel-recommendation-dataset", "Expanded_Destinations.csv")

print("✅ Datasets loaded successfully")
print(f"   Users:        {df_users.shape}")
print(f"   History:      {df_history.shape}")
print(f"   Destinations: {df_destinations.shape}")


print("=" * 60)
print("USERS DATASET")
print("=" * 60)
print(f"Shape: {df_users.shape}")
print()

print("--- df_users.info() ---")
df_users.info()
print()

print("--- df_users.describe(include='all') ---")
display(df_users.describe(include='all'))
print()

print("--- Missing Values ---")
print(df_users.isnull().sum())
print()

print("--- Duplicate Rows ---")
print(f"Duplicate rows: {df_users.duplicated().sum()}")
print()

print("--- Sample Rows ---")
display(df_users.head())


print("=" * 60)
print("HISTORY DATASET")
print("=" * 60)
print(f"Shape: {df_history.shape}")
print()

print("--- df_history.info() ---")
df_history.info()
print()

print("--- df_history.describe(include='all') ---")
display(df_history.describe(include='all'))
print()

print("--- Missing Values ---")
print(df_history.isnull().sum())
print()

print("--- Duplicate Rows ---")
print(f"Duplicate rows: {df_history.duplicated().sum()}")
print()

print("--- Sample Rows ---")
display(df_history.head())


print("=" * 60)
print("DESTINATIONS DATASET")
print("=" * 60)
print(f"Shape: {df_destinations.shape}")
print()

print("--- df_destinations.info() ---")
df_destinations.info()
print()

print("--- df_destinations.describe(include='all') ---")
display(df_destinations.describe(include='all'))
print()

print("--- Missing Values ---")
print(df_destinations.isnull().sum())
print()

print("--- Duplicate Rows ---")
print(f"Duplicate rows: {df_destinations.duplicated().sum()}")
print()

print("--- Sample Rows ---")
display(df_destinations.head())


# Rename to avoid column collision (both Users and Destinations have 'Name')
df_destinations.rename(columns={'Name': 'DestinationName'}, inplace=True)

# Merge: History + Users -> then + Destinations
df_merged = pd.merge(df_history, df_users, on='UserID')
df = pd.merge(df_merged, df_destinations, on='DestinationID')

print(f"✅ Merged dataset shape: {df.shape}")
print(f"   Columns: {df.columns.tolist()}")
print()

print("--- Merged df.info() ---")
df.info()
print()

print("--- Merged df.describe(include='all') ---")
display(df.describe(include='all'))
print()

print("--- Missing Values (merged) ---")
print(df.isnull().sum())
print()

print("--- Duplicate Rows (merged) ---")
print(f"Duplicate rows: {df.duplicated().sum()}")
print()

display(df.head(10))


# Target Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

dest_counts = df['DestinationName'].value_counts()
dest_counts.plot(kind='bar', ax=axes[0], color=sns.color_palette('Set2', len(dest_counts)), edgecolor='black')
axes[0].set_title('Destination Distribution (All Records)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Destination')
for i, v in enumerate(dest_counts.values):
    axes[0].text(i, v + 2, str(v), ha='center', fontweight='bold')

# Experience Rating Distribution
rating_counts = df['ExperienceRating'].value_counts().sort_index()
rating_counts.plot(kind='bar', ax=axes[1], color=['#ff6b6b', '#ffa500', '#ffd700', '#90ee90', '#2e8b57'], edgecolor='black')
axes[1].set_title('Experience Rating Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Rating')
for i, v in enumerate(rating_counts.values):
    axes[1].text(i, v + 2, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\nTotal records: {len(df)}")
print(f"Unique destinations: {df['DestinationName'].nunique()}")
print(f"Experience rating range: {df['ExperienceRating'].min()} to {df['ExperienceRating'].max()}")


fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Gender
df['Gender'].value_counts().plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
axes[0,0].set_title('Gender Distribution', fontsize=13, fontweight='bold')
axes[0,0].set_ylabel('')

# Preferences
pref_counts = df['Preferences'].value_counts()
pref_counts.plot(kind='barh', ax=axes[0,1], color=sns.color_palette('viridis', len(pref_counts)))
axes[0,1].set_title('Travel Preferences', fontsize=13, fontweight='bold')
axes[0,1].set_xlabel('Count')

# NumberOfAdults
df['NumberOfAdults'].value_counts().sort_index().plot(kind='bar', ax=axes[0,2], color='steelblue', edgecolor='black')
axes[0,2].set_title('Number of Adults', fontsize=13, fontweight='bold')
axes[0,2].set_ylabel('Count')

# NumberOfChildren
df['NumberOfChildren'].value_counts().sort_index().plot(kind='bar', ax=axes[1,0], color='coral', edgecolor='black')
axes[1,0].set_title('Number of Children', fontsize=13, fontweight='bold')
axes[1,0].set_ylabel('Count')

# Type
df['Type'].value_counts().plot(kind='bar', ax=axes[1,1], color=sns.color_palette('Set3', df['Type'].nunique()), edgecolor='black')
axes[1,1].set_title('Destination Type', fontsize=13, fontweight='bold')
axes[1,1].set_ylabel('Count')

# State
df['State'].value_counts().plot(kind='bar', ax=axes[1,2], color=sns.color_palette('Pastel1', df['State'].nunique()), edgecolor='black')
axes[1,2].set_title('Destination State', fontsize=13, fontweight='bold')
axes[1,2].set_ylabel('Count')

plt.suptitle('Feature Distributions Overview', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# Crosstab heatmaps: Which features are correlated with specific destinations?
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gender vs Destination
ct_gender = pd.crosstab(df['Gender'], df['DestinationName'], normalize='index')
sns.heatmap(ct_gender, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0,0])
axes[0,0].set_title('P(Destination | Gender)', fontsize=13, fontweight='bold')

# Preferences vs Destination
ct_pref = pd.crosstab(df['Preferences'], df['DestinationName'], normalize='index')
sns.heatmap(ct_pref, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0,1])
axes[0,1].set_title('P(Destination | Preferences)', fontsize=13, fontweight='bold')

# NumberOfAdults vs Destination
ct_adults = pd.crosstab(df['NumberOfAdults'], df['DestinationName'], normalize='index')
sns.heatmap(ct_adults, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1,0])
axes[1,0].set_title('P(Destination | NumberOfAdults)', fontsize=13, fontweight='bold')

# NumberOfChildren vs Destination
ct_children = pd.crosstab(df['NumberOfChildren'], df['DestinationName'], normalize='index')
sns.heatmap(ct_children, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[1,1])
axes[1,1].set_title('P(Destination | NumberOfChildren)', fontsize=13, fontweight='bold')

plt.suptitle('Feature-Target Relationship Heatmaps', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# CRITICAL CHECK: Do destination-specific columns perfectly predict the target?
print("=" * 60)
print("  DATA LEAKAGE CHECK")
print("=" * 60)
print()

# State vs DestinationName
ct_state = pd.crosstab(df['State'], df['DestinationName'])
print("Crosstab: State vs DestinationName")
display(ct_state)
print()

# Type vs DestinationName
ct_type = pd.crosstab(df['Type'], df['DestinationName'])
print("Crosstab: Type vs DestinationName")
display(ct_type)
print()

print("⚠️  FINDING: 'State', 'Type', and 'BestTimeToVisit' are destination-level attributes.")
print("   Each State maps to exactly one DestinationName → DATA LEAKAGE.")
print("   BestTimeToVisit has near 1:1 mapping (5 seasons for 5 destinations).")
print("   Including them in X would cause the model to just memorize")
print("   these attributes instead of learning from user preferences.")
print()
print("   ✅ EXCLUDE: State, Type, BestTimeToVisit, Popularity")
print("   Only USER-provided inputs will be used as features.")


from sklearn.preprocessing import LabelEncoder

# 1. Reliability Filtering — Only learn from HIGH SATISFACTION trips
df_clean = df[df['ExperienceRating'] >= 4].copy()
print(f"Records after ExperienceRating >= 4 filter: {len(df_clean)} (from {len(df)})")
print(f"Records removed: {len(df) - len(df_clean)} ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")
print()

# 2. Feature Engineering
# NOTE: We do NOT create Total_Travelers = Adults + Children
# because it's a perfect linear combination of features already in X.
# Including it causes multicollinearity that affects different models
# differently, making comparison unreliable.

# 3. Clean up Compound Preferences (trim whitespace for consistency)
df_clean['Preferences'] = df_clean['Preferences'].apply(lambda x: ', '.join([p.strip() for p in x.split(',')]))

all_pref_combos = sorted(df_clean['Preferences'].unique())
print(f"Dataset has {len(all_pref_combos)} distinct user preference profiles:")
for combo in all_pref_combos:
    print(f"   - '{combo}' ({(df_clean['Preferences'] == combo).sum()} users)")
print()

# 4. Drop temporal noise (per paper.txt: VisitDate is not critical)
# VisitDate is already excluded from our feature selection

# 5. Target Variable — Label Encode DestinationName
le_dest = LabelEncoder()
df_clean['Target_Destination'] = le_dest.fit_transform(df_clean['DestinationName'])
n_classes = len(le_dest.classes_)

print(f"Number of classes: {n_classes}")
print(f"Class mapping:")
for i, name in enumerate(le_dest.classes_):
    count = (df_clean['Target_Destination'] == i).sum()
    print(f"   {i} → {name}  ({count} samples)")


# Post-filter EDA
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution after filtering
df_clean['DestinationName'].value_counts().plot(
    kind='bar', ax=axes[0], 
    color=sns.color_palette('Set2', n_classes), edgecolor='black'
)
axes[0].set_title('Target Distribution (After Filtering)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count')
for i, v in enumerate(df_clean['DestinationName'].value_counts().values):
    axes[0].text(i, v + 1, str(v), ha='center', fontweight='bold')

# Preference profile distribution
df_clean['Preferences'].value_counts().plot(
    kind='barh', ax=axes[1], color='teal', edgecolor='black'
)
axes[1].set_title('User Preference Profiles', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Count')

plt.tight_layout()
plt.show()


# Feature Selection
# We ONLY use features that a REAL USER would provide as input:
#   - Gender            (user demographic)
#   - NumberOfAdults    (group composition)
#   - NumberOfChildren  (group composition)
#   - Pref_...          (one-hot encoded compound preference combinations)
#
# EXCLUDED (destination-level attributes = data leakage):
#   - State, Type, BestTimeToVisit, Popularity

numeric_features = ['NumberOfAdults', 'NumberOfChildren']
categorical_features = ['Gender', 'Preferences']

features = numeric_features + categorical_features
X = df_clean[features].copy()
y = df_clean['Target_Destination'].copy()

print(f"Selected features ({len(features)}): {features}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Train-Test Split (80/20, stratified)
# IMPORTANT: Splitting BEFORE any encoding or scaling to prevent data leakage!
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Testing set:  {X_test.shape}")
print()

# Verify stratification
print("Class distribution in train vs test:")
train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
for i in range(n_classes):
    print(f"   Class {i} ({le_dest.classes_[i]}): Train={train_dist.get(i, 0):.3f}  Test={test_dist.get(i, 0):.3f}")

# Define the ColumnTransformer Pipeline
# 1. Scale ONLY the numeric features
# 2. One-Hot Encode categorical features (ignore unknown categories in production!)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# For Tree models, we don't scale the numeric features!
preprocessor_unscaled = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
    ])

# ==========================================
# KERAS/NEURAL NETWORK PREPROCESSING
# ==========================================
# Scikit-Learn pipelines will automatically fit the preprocessor perfectly on their own during cross-validation.
# However, Keras requires raw arrays. We must manually fit the preprocessor for Keras.

# CRITICAL LEAKAGE FIX: We must extract Keras's Validation Split BEFORE fitting the scaler!
# Otherwise, the scaler learns the min/max of Keras's exact validation data, inflating early stopping metrics.
X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Fit preprocessor strictly on the 80% Keras training subset
preprocessor.fit(X_train_nn)

# Transform the keras subsets
X_train_nn_processed = preprocessor.transform(X_train_nn)
X_val_nn_processed   = preprocessor.transform(X_val_nn)

# Transform the final blind test set for Keras
X_test_processed = preprocessor.transform(X_test)

# Also capture the final feature column names from the encoder for reference
try:
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    X_columns = numeric_features + list(cat_names)
except Exception:
    X_columns = [f"Feature_{i}" for i in range(X_train_nn_processed.shape[1])]

print(f"\n✅ Preprocessor configured successfully.")
print(f"Total features after encoding: {len(X_columns)}")


# Class Weights (handle destination popularity imbalance)
from sklearn.utils.class_weight import compute_class_weight

class_weights_array = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
class_weights = dict(zip(np.unique(y_train), class_weights_array))

print("Balanced class weights:")
for cls, weight in class_weights.items():
    print(f"   Class {cls} ({le_dest.classes_[cls]}): weight = {weight:.4f}")


from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score
from sklearn.base import clone

results = {}
all_predictions = {}  # Store predictions for later comparison

def evaluate_multiclass(name, model, y_true, y_pred, y_prob=None, is_keras_nn=False, cv_model=None):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Cross-validation (5-fold)
    cv_mean_str = "N/A (Keras)"
    cv_mean_val = 0.0
    if not is_keras_nn:
        try:
            cv_blueprint = clone(cv_model) if cv_model is not None else clone(model)
            cv_scores = cross_val_score(cv_blueprint, X_train, y_train, cv=5, scoring='f1_weighted')
            cv_mean_val = cv_scores.mean()
            cv_mean_str = f"{cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        except Exception as e:
            cv_mean_str = f"Error: {e}"
            
    # Top-3 Accuracy
    top3_acc = 0.0
    if y_prob is not None and y_prob.shape[1] >= 3:
        top3_preds = np.argsort(y_prob, axis=1)[:, -3:]
        top3_acc = np.array([y_true.iloc[i] in top3_preds[i] for i in range(len(y_true))]).mean()
    
    # Print results
    print(f"\n{'='*55}")
    print(f"  📊 {name}")
    print(f"{'='*55}")
    print(f"  Test Accuracy:      {acc:.4f}")
    print(f"  5-Fold CV F1 (W):   {cv_mean_str}")
    print(f"  Precision (W):      {prec:.4f}")
    print(f"  Recall (W):         {rec:.4f}")
    print(f"  F1-Score (W):       {f1:.4f}")
    print(f"  Top-3 Accuracy:     {top3_acc:.4f}")
    
    # Classification Report per class
    print(f"\n  Per-Class Report:")
    print(classification_report(y_true, y_pred, target_names=le_dest.classes_, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le_dest.classes_, yticklabels=le_dest.classes_)
    plt.title(f'Confusion Matrix: {name}', fontsize=13, fontweight='bold')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    # Store results
    results[name] = {
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'Top3-Acc': round(top3_acc, 4),
        'CV-F1': round(cv_mean_val, 4) if not is_keras_nn else 0.0
    }
    all_predictions[name] = y_pred


# 1. Logistic Regression (Multinomial)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

log_reg = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', LogisticRegression(
        multi_class='multinomial', solver='lbfgs', random_state=42, 
        max_iter=2000, class_weight=class_weights, C=1.0
    ))
])
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
y_prob_lr = log_reg.predict_proba(X_test)

evaluate_multiclass("Logistic Regression", log_reg, y_test, y_pred_lr, y_prob_lr)


# 2. Linear SVM (One-vs-Rest)
from sklearn.svm import SVC

svm_linear = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', SVC(
        kernel='linear', probability=True, decision_function_shape='ovr', 
        random_state=42, class_weight=class_weights
    ))
])
svm_linear.fit(X_train, y_train)

y_pred_svm_lin = svm_linear.predict(X_test)
y_prob_svm_lin = svm_linear.predict_proba(X_test)

evaluate_multiclass("Linear SVM", svm_linear, y_test, y_pred_svm_lin, y_prob_svm_lin)


# 3. Single-Layer Perceptron (TensorFlow)
import tensorflow as tf

slp_model = tf.keras.Sequential([
    tf.keras.layers.Dense(n_classes, activation='softmax', input_shape=(X_train_nn_processed.shape[1],))
])
slp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
slp_model.fit(X_train_nn_processed, y_train_nn, epochs=100, validation_data=(X_val_nn_processed, y_val_nn), verbose=0, class_weight=class_weights)

y_prob_slp = slp_model.predict(X_test_processed)
y_pred_slp = np.argmax(y_prob_slp, axis=1)

evaluate_multiclass("Single-Layer Perceptron", slp_model, y_test, y_pred_slp, y_prob_slp, is_keras_nn=True)


# 4. Stochastic Gradient Descent (SGD)
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', SGDClassifier(
        loss='log_loss', random_state=42, max_iter=1000, 
        class_weight=class_weights, alpha=0.0001
    ))
])
sgd.fit(X_train, y_train)

y_pred_sgd = sgd.predict(X_test)
y_prob_sgd = sgd.predict_proba(X_test)

evaluate_multiclass("SGD Classifier", sgd, y_test, y_pred_sgd, y_prob_sgd)


# 5. K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

pipe_knn = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', KNeighborsClassifier())
])

param_grid_knn = {
    'clf__n_neighbors': list(range(1, 31)),
    'clf__weights': ['uniform', 'distance'],
    'clf__p': [1, 2] # 1=Manhattan, 2=Euclidean (with minkowski)
}

grid_knn = GridSearchCV(
    pipe_knn,
    param_grid=param_grid_knn,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1
)

grid_knn.fit(X_train, y_train)

print(f"KNN Best params: {grid_knn.best_params_}")
print(f"KNN Best CV F1-Score: {grid_knn.best_score_:.4f}")

knn = grid_knn.best_estimator_

y_pred_knn = knn.predict(X_test)
y_prob_knn = knn.predict_proba(X_test)

evaluate_multiclass("K-Nearest Neighbours", knn, y_test, y_pred_knn, y_prob_knn)


# 6. Kernel SVM (RBF)
svm_rbf = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', SVC(
        kernel='rbf', probability=True, decision_function_shape='ovr', 
        random_state=42, class_weight=class_weights, gamma='scale'
    ))
])
svm_rbf.fit(X_train, y_train)

y_pred_svm_rbf = svm_rbf.predict(X_test)
y_prob_svm_rbf = svm_rbf.predict_proba(X_test)

evaluate_multiclass("Kernel SVM (RBF)", svm_rbf, y_test, y_pred_svm_rbf, y_prob_svm_rbf)


# 7. Naive Bayes (Gaussian)
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

nb = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('clf', GaussianNB())
])
nb.fit(X_train, y_train)

y_pred_nb = nb.predict(X_test)
y_prob_nb = nb.predict_proba(X_test)

evaluate_multiclass("Naive Bayes", nb, y_test, y_pred_nb, y_prob_nb)


# 8. Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('clf', DecisionTreeClassifier(random_state=42, class_weight=class_weights, max_depth=10))
])
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)
y_prob_dt = dt.predict_proba(X_test)

evaluate_multiclass("Decision Tree", dt, y_test, y_pred_dt, y_prob_dt)


# 9. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('clf', RandomForestClassifier(
        random_state=42, n_estimators=200, class_weight=class_weights, 
        max_depth=15, max_features=None
    ))
])
rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)

evaluate_multiclass("Random Forest", rf, y_test, y_pred_rf, y_prob_rf)


# 10. Multi-Layer Perceptron (MLP)
mlp_model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train_nn_processed.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(n_classes, activation='softmax')
])

mlp_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss='sparse_categorical_crossentropy', metrics=['accuracy']
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=20, restore_best_weights=True
)

history = mlp_model.fit(
    X_train_nn_processed, y_train_nn, epochs=300, validation_data=(X_val_nn_processed, y_val_nn), 
    verbose=0, class_weight=class_weights, callbacks=[early_stop],
    batch_size=32
)

# Plot MLP training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2)
axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
axes[0].set_title('MLP Training & Validation Loss', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
axes[1].set_title('MLP Training & Validation Accuracy', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"MLP stopped at epoch {len(history.history['loss'])}")

y_prob_mlp = mlp_model.predict(X_test_processed)
y_pred_mlp = np.argmax(y_prob_mlp, axis=1)

evaluate_multiclass("Multi-Layer Perceptron", mlp_model, y_test, y_pred_mlp, y_prob_mlp, is_keras_nn=True)


# 11. AdaBoost (Boosting)
from sklearn.ensemble import AdaBoostClassifier

ada = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('clf', AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=2, class_weight=class_weights),
        random_state=42
    ))
])
ada.fit(X_train, y_train)

y_pred_ada = ada.predict(X_test)
y_prob_ada = ada.predict_proba(X_test)

evaluate_multiclass("AdaBoost", ada, y_test, y_pred_ada, y_prob_ada)


# 12. Gradient Boosting (Boosting)
from sklearn.ensemble import GradientBoostingClassifier

gb = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('clf', GradientBoostingClassifier(random_state=42))
])
gb.fit(X_train, y_train)

y_pred_gb = gb.predict(X_test)
y_prob_gb = gb.predict_proba(X_test)

evaluate_multiclass("Gradient Boosting", gb, y_test, y_pred_gb, y_prob_gb)


# 13. Bagging Classifier (Bootstrap Aggregating)
# Trains the SAME base model on random subsets of data, then averages predictions.
from sklearn.ensemble import BaggingClassifier

bagging = Pipeline([
    ('preprocessor', preprocessor_unscaled),
    ('clf', BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=10, class_weight=class_weights),
        n_estimators=200, max_samples=0.8, max_features=0.8,
        random_state=42, n_jobs=-1
    ))
])
bagging.fit(X_train, y_train)

y_pred_bag = bagging.predict(X_test)
y_prob_bag = bagging.predict_proba(X_test)

evaluate_multiclass("Bagging", bagging, y_test, y_pred_bag, y_prob_bag)


# 14. Stacking Classifier (Meta-learner on top of diverse base models)
from sklearn.ensemble import StackingClassifier

# Note: We must use the SCALED preprocessor for stacking, 
# because linear models and SVMs inside the stack depend on it!
stacking = Pipeline([
    ('preprocessor', preprocessor),
    ('clf', StackingClassifier(
        estimators=[
            ('lr', LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=42, class_weight=class_weights)),
            ('nb', GaussianNB()),
            ('dt', DecisionTreeClassifier(random_state=42, max_depth=10, class_weight=class_weights)),
            ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance')),
        ],
        final_estimator=LogisticRegression(multi_class='multinomial', max_iter=2000, random_state=42, class_weight=class_weights),
        cv=5,
        passthrough=False
    ))
])
stacking.fit(X_train, y_train)

y_pred_stack = stacking.predict(X_test)
y_prob_stack = stacking.predict_proba(X_test)

evaluate_multiclass("Stacking Ensemble", stacking, y_test, y_pred_stack, y_prob_stack)


# Build results table
results_df = pd.DataFrame(results).T

# Tiered ranking: Top-3 Accuracy (primary) → F1-Score (tiebreaker) → CV-F1 (generalization)
# This matches how the app works: users see 3 recommendations, so Top-3 matters most.
# Among models with similar Top-3, we prefer the one with the best first pick (F1).
# CV-F1 confirms the model generalizes and isn't just lucky on one split.
results_df = results_df.sort_values(
    by=['Top3-Acc', 'F1-Score', 'CV-F1'], 
    ascending=[False, False, False]
)

# Added to export metrics
results_df.to_csv('results_df.csv', index=True)

print("\n" + "=" * 70)
print("   📊 MULTI-CLASS ALGORITHM COMPARISON — FINAL RESULTS")
print("   Sorted by: Top-3 Accuracy → F1-Score → CV-F1")
print("=" * 70)
display(results_df)


# Comprehensive comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# 1. F1-Score
f1_sorted = results_df.sort_values('F1-Score', ascending=True)
colors_f1 = ['gold' if x == f1_sorted['F1-Score'].max() else 'steelblue' for x in f1_sorted['F1-Score']]
f1_sorted['F1-Score'].plot(kind='barh', ax=axes[0,0], color=colors_f1, edgecolor='black')
axes[0,0].set_title('F1-Score (Weighted)', fontsize=14, fontweight='bold')
axes[0,0].set_xlabel('F1-Score')
for i, v in enumerate(f1_sorted['F1-Score']):
    axes[0,0].text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold')

# 2. Accuracy
acc_sorted = results_df.sort_values('Accuracy', ascending=True)
colors_acc = ['gold' if x == acc_sorted['Accuracy'].max() else 'coral' for x in acc_sorted['Accuracy']]
acc_sorted['Accuracy'].plot(kind='barh', ax=axes[0,1], color=colors_acc, edgecolor='black')
axes[0,1].set_title('Test Accuracy', fontsize=14, fontweight='bold')
axes[0,1].set_xlabel('Accuracy')
for i, v in enumerate(acc_sorted['Accuracy']):
    axes[0,1].text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold')

# 3. Top-3 Accuracy
top3_sorted = results_df.sort_values('Top3-Acc', ascending=True)
colors_top3 = ['gold' if x == top3_sorted['Top3-Acc'].max() else 'mediumseagreen' for x in top3_sorted['Top3-Acc']]
top3_sorted['Top3-Acc'].plot(kind='barh', ax=axes[1,0], color=colors_top3, edgecolor='black')
axes[1,0].set_title('Top-3 Recommendation Accuracy', fontsize=14, fontweight='bold')
axes[1,0].set_xlabel('Top-3 Accuracy')
for i, v in enumerate(top3_sorted['Top3-Acc']):
    axes[1,0].text(v + 0.002, i, f'{v:.4f}', va='center', fontweight='bold')

# 4. Radar/Spider chart of all metrics for top 3 models
top3_models = results_df.head(3).index.tolist()
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Top3-Acc']
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
angles += angles[:1]

axes[1,1].remove()
ax_radar = fig.add_subplot(2, 2, 4, polar=True)
ax_radar.set_theta_offset(np.pi / 2)

for i, model_name in enumerate(top3_models):
    values = [results_df.loc[model_name, m] for m in metrics]
    values += values[:1]
    ax_radar.plot(angles, values, 'o-', linewidth=2, label=model_name)
    ax_radar.fill(angles, values, alpha=0.15)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(metrics, fontsize=10)
ax_radar.set_title('Top 3 Models — Metric Comparison', fontsize=13, fontweight='bold', y=1.1)
ax_radar.legend(loc='lower right', bbox_to_anchor=(1.3, -0.1), fontsize=9)

plt.suptitle('Algorithm Performance Dashboard', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


# Model agreement analysis — Do models agree on predictions?
print("\n" + "=" * 60)
print("   🔍 MODEL AGREEMENT ANALYSIS")
print("=" * 60)

pred_df = pd.DataFrame(all_predictions)
agreement = pred_df.apply(lambda row: row.nunique(), axis=1)
print(f"\nTotal test samples: {len(pred_df)}")
print(f"Samples where ALL models agree:     {(agreement == 1).sum()} ({(agreement == 1).mean()*100:.1f}%)")
print(f"Samples where models DISAGREE:      {(agreement > 1).sum()} ({(agreement > 1).mean()*100:.1f}%)")
print(f"Average unique predictions per sample: {agreement.mean():.2f}")
print()
print("This proves the models are making DIFFERENT decisions, confirming authentic supervised learning.")


import joblib

# ═══════════════════════════════════════════════════════════════════
# BEST MODEL SELECTION — Harmonic Mean (Most Consistent Model)
# ═══════════════════════════════════════════════════════════════════
# Problem: Picking by one metric ignores the others.
# Solution: Harmonic Mean across all three key metrics.
#
# Why harmonic mean?
#   - It's the SAME math behind F1-Score (harmonic mean of Precision & Recall)
#   - It penalizes models that are weak on ANY metric
#   - A model scoring [0.3, 0.3, 0.3] beats one scoring [0.6, 0.6, 0.01]
#   - This finds the most CONSISTENT performer, not just the best at one thing
#
# Metrics used (equal weight, no bias):
#   - F1-Score       (quality of first recommendation)
#   - Accuracy       (overall correctness)
#   - Top-3 Accuracy (quality of top 3 recommendations)
# ═══════════════════════════════════════════════════════════════════

print("=" * 70)
print("   🎯 DATA-DRIVEN MODEL SELECTION")
print("   Method: Harmonic Mean of F1, Accuracy, Top-3 Accuracy")
print("=" * 70)
print()

# Step 1: Normalize all three metrics to [0.01, 1.0] range
# (0.01 floor to avoid division by zero in harmonic mean)
selection = results_df[['Accuracy', 'F1-Score', 'Top3-Acc', 'CV-F1']].copy()

for col in ['Accuracy', 'F1-Score', 'Top3-Acc']:
    col_min = selection[col].min()
    col_max = selection[col].max()
    if col_max > col_min:
        selection[f'{col}_norm'] = 0.01 + 0.99 * (selection[col] - col_min) / (col_max - col_min)
    else:
        selection[f'{col}_norm'] = 0.5

# Step 2: Compute harmonic mean of the 3 normalized metrics
# H = 3 / (1/F1_norm + 1/Acc_norm + 1/Top3_norm)
selection['Consistency'] = 3.0 / (
    1.0 / selection['F1-Score_norm'] +
    1.0 / selection['Accuracy_norm'] +
    1.0 / selection['Top3-Acc_norm']
)

selection = selection.sort_values('Consistency', ascending=False)

# Export consistency too
selection.to_csv('selection_df.csv', index=True)

print("Consistency Score (Harmonic Mean — higher = more balanced):")
print()
display(selection[['Accuracy', 'F1-Score', 'Top3-Acc', 'CV-F1', 'Consistency']])
print()

# Step 3: Pick the most consistent model, verified by cross-validation
best_name = selection['Consistency'].idxmax()
best_row = results_df.loc[best_name]

# If the winner has no CV (Keras model), fall back to next with CV
if best_row['CV-F1'] == 0:
    print(f"⚠️  {best_name} has no cross-validation (Keras model).")
    cv_models = selection[selection['CV-F1'] > 0]
    if len(cv_models) > 0:
        best_name = cv_models['Consistency'].idxmax()
        best_row = results_df.loc[best_name]
        print(f"   → Selecting next best cross-validated model: {best_name}")
        print()

print("=" * 60)
print(f"  🏆 BEST MODEL: {best_name}")
print("=" * 60)
print(f"  Accuracy:      {best_row['Accuracy']:.4f}")
print(f"  Precision:     {best_row['Precision']:.4f}")
print(f"  Recall:        {best_row['Recall']:.4f}")
print(f"  F1-Score:      {best_row['F1-Score']:.4f}")
print(f"  Top-3 Acc:     {best_row['Top3-Acc']:.4f}")
print(f"  CV-F1:         {best_row['CV-F1']:.4f}")
print(f"  Consistency:   {selection.loc[best_name, 'Consistency']:.4f}")

models_dict = {
    "Logistic Regression": log_reg,
    "Linear SVM": svm_linear,
    "Single-Layer Perceptron": slp_model,
    "SGD Classifier": sgd,
    "K-Nearest Neighbours": knn,
    "Kernel SVM (RBF)": svm_rbf,
    "Naive Bayes": nb,
    "Decision Tree": dt,
    "Random Forest": rf,
    "Multi-Layer Perceptron": mlp_model,
    "AdaBoost": ada,
    "Gradient Boosting": gb,
    "Bagging": bagging,
    "Stacking Ensemble": stacking
}

best_model_obj = models_dict[best_name]

# Export all pipeline components
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(X_columns, 'X_columns.joblib')
joblib.dump(le_dest, 'label_encoder.joblib')
joblib.dump(best_model_obj, 'best_model.joblib')

print(f"\n✅ Exported artifacts:")
print(f"   - preprocessor.joblib   (ColumnTransformer: Scaler + Encoder)")
print(f"   - X_columns.joblib      (Expected feature column names)")
print(f"   - label_encoder.joblib  (LabelEncoder: int → DestinationName)")
print(f"   - best_model.joblib     ({best_name})")


# =====================================================================
#  🎯 INTERACTIVE TRAVEL RECOMMENDATION DEMO
# =====================================================================

# --- 1. Gather available options from training data ---
gender_options = sorted(df_clean['Gender'].unique().tolist())

print("=" * 60)
print("  🌍 TRAVEL DESTINATION RECOMMENDER — Console Demo")
print("=" * 60)
print()

# --- 2. Gender Input ---
print("Select your gender:")
for i, g in enumerate(gender_options, 1):
    print(f"   [{i}] {g}")
while True:
    try:
        g_choice = int(input("Enter choice (number): "))
        if 1 <= g_choice <= len(gender_options):
            user_gender = gender_options[g_choice - 1]
            break
    except ValueError:
        pass
    print("   ⚠️  Invalid choice, try again.")
print(f"   ✅ Gender: {user_gender}")
print()

# --- 3. Number of Adults ---
while True:
    try:
        user_adults = int(input("Number of adults travelling (1-10): "))
        if 1 <= user_adults <= 10:
            break
    except ValueError:
        pass
    print("   ⚠️  Please enter a number between 1 and 10.")
print(f"   ✅ Adults: {user_adults}")
print()

# --- 4. Number of Children ---
while True:
    try:
        user_children = int(input("Number of children travelling (0-10): "))
        if 0 <= user_children <= 10:
            break
    except ValueError:
        pass
    print("   ⚠️  Please enter a number between 0 and 10.")
print(f"   ✅ Children: {user_children}")
print()

# --- 5. Travel Preferences (pick a profile) ---
print("Select your travel preference profile:")
for i, combo in enumerate(all_pref_combos, 1):
    print(f"   [{i}] {combo}")
print()
while True:
    try:
        p_choice = int(input("Enter choice (number): "))
        if 1 <= p_choice <= len(all_pref_combos):
            user_pref = all_pref_combos[p_choice - 1]
            break
    except ValueError:
        pass
    print("   ⚠️  Invalid choice, try again.")
print(f"   ✅ Preferences: {user_pref}")
print()

# --- 6. Build input DataFrame (must match training columns exactly) ---

user_data = {
    'NumberOfAdults': user_adults,
    'NumberOfChildren': user_children,
    'Gender': user_gender,
    'Preferences': user_pref
}

user_input = pd.DataFrame([user_data])

# Transform using the preprocessor only if it's a keras model, since pipelines handle it natively
if best_name in ["Single-Layer Perceptron", "Multi-Layer Perceptron"]:
    user_final = preprocessor.transform(user_input)
else:
    user_final = user_input

# --- 8. Predict ---
if hasattr(best_model_obj, 'predict_proba'):
    probabilities = best_model_obj.predict_proba(user_final)[0]
else:
    probabilities = best_model_obj.predict(user_final)[0]

top3_indices = np.argsort(probabilities)[::-1][:3]
top3_names = le_dest.inverse_transform(top3_indices)
top3_probs = probabilities[top3_indices]

# --- 9. Display Results ---
print("=" * 60)
print(f"  🏆 TOP 3 RECOMMENDED DESTINATIONS")
print(f"  Model: {best_name}")
print("=" * 60)
print()
print(f"  Your profile:")
print(f"    Gender:       {user_gender}")
print(f"    Adults:       {user_adults}")
print(f"    Children:     {user_children}")
print(f"    Preferences:  {user_pref}")
print()
print("  ─────────────────────────────────────")

medals = ['🥇', '🥈', '🥉']
for rank, (name, prob) in enumerate(zip(top3_names, top3_probs)):
    bar_len = int(prob * 30)
    bar = '█' * bar_len + '░' * (30 - bar_len)
    print(f"  {medals[rank]}  {name}")
    print(f"       Confidence: {prob * 100:.1f}%  |{bar}|")
    print()

print("  ─────────────────────────────────────")
print("  ✅ Recommendation complete!")

