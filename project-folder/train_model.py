# train_model.py

import pandas as pd
import pickle
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier
import numpy as np

# Import custom_preprocessor from preprocessing.py
from preprocessing import custom_preprocessor

# Load the dataset
df = pd.read_excel('files/RASFF_data_Preprocess1.xlsx', sheet_name='Sheet1')

# Filter out rows with 'undecided' risk decision
df = df[df['risk_decision'] != 'undecided']

# Select relevant features and target
features = df[['category', 'type', 'subject', 'notifying_country', 'classification',
               'operator', 'origin', 'hazards']].copy()
target = df['risk_decision']

# Encode the target labels
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save unique values for drop-down menus
dropdown_options = {
    'category': sorted(features['category'].unique().tolist()),
    'type': sorted(features['type'].unique().tolist()),
    'notifying_country': sorted(features['notifying_country'].unique().tolist()),
    'classification': sorted(features['classification'].unique().tolist())
}

# Save the dropdown options to a JSON file
with open('dropdown_options.json', 'w') as f:
    json.dump(dropdown_options, f)

# Define TF-IDF vectorizers for text columns
text_transformer = TfidfVectorizer(
    max_features=500,
    stop_words='english',
    preprocessor=custom_preprocessor
)
hazards_tfidf = TfidfVectorizer(
    max_features=150,
    stop_words='english',
    preprocessor=custom_preprocessor
)
operator_tfidf = TfidfVectorizer(
    max_features=150,
    stop_words='english',
    preprocessor=custom_preprocessor
)
origin_tfidf = TfidfVectorizer(
    max_features=150,
    stop_words='english',
    preprocessor=custom_preprocessor
)

# OneHotEncoder for categorical columns
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# ColumnTransformer to apply transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('text', text_transformer, 'subject'),
        ('hazards_tfidf', hazards_tfidf, 'hazards'),
        ('operator_tfidf', operator_tfidf, 'operator'),
        ('origin_tfidf', origin_tfidf, 'origin'),
        ('cat', categorical_transformer, ['category', 'type', 'notifying_country', 'classification'])
    ]
)

# Define XGBoost model
xgb_model = XGBClassifier(
    eval_metric='mlogloss',
    random_state=42,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)

# Create the pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# Train the model using Stratified K-Fold Cross-Validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

fold = 1
for train_index, test_index in skf.split(features, target_encoded):
    X_train, X_test = features.iloc[train_index], features.iloc[test_index]
    y_train, y_test = target_encoded[train_index], target_encoded[test_index]

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"Fold {fold} Accuracy: {accuracy:.2f}")
    print(f"Fold {fold} Macro F1-Score: {macro_f1:.2f}")
    fold += 1

# Train the model on the entire dataset
pipeline.fit(features, target_encoded)

# Save the trained pipeline
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

print("Model training complete. Model pipeline, label encoder, and dropdown options saved successfully.")
