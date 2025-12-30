import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

DATA_FILE = "data/gestures.csv"
MODEL_PATH = "models/gesture_model.pkl"
SCALER_PATH = "models/scaler.pkl"

def train_model():
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file {DATA_FILE} not found. Run src.collect first.")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # Basic cleaning
    df = df.dropna()
    if len(df) < 50:
        print("Error: Not enough data to train (need > 50 samples).")
        return

    print(f"Total samples: {len(df)}")
    print(df['label'].value_counts())

    X = df.drop(columns=['timestamp', 'label']).values
    y = df['label'].values

    # Train/Val Split
    # We use a simple time-based split assumption or random split.
    # Random split is safer if users record classes in blocks.
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling (Optional but good practice)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("Training Random Forest...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # Evaluation
    print("\nValidation Results:")
    y_pred = clf.predict(X_val_scaled)
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    # Save
    print(f"Saving model to {MODEL_PATH}...")
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Done.")

if __name__ == "__main__":
    train_model()
