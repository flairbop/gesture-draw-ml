import pandas as pd
import numpy as np
import joblib
import os
import json
import argparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_FILE = "data/gestures.csv"
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.pkl")
LABEL_ORDER_PATH = os.path.join(MODELS_DIR, "label_order.json")

def train_model(model_type='logreg'):
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Run src.collect first.")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)
    
    # 1. Clean
    initial_len = len(df)
    df = df.dropna()
    dropped = initial_len - len(df)
    if dropped > 0:
        print(f"Dropped {dropped} rows with NaNs.")
        
    if len(df) < 100:
        print("Error: Not enough data (<100 samples).")
        return

    # 2. Quality Report
    print("\n--- Data Quality Report ---")
    print("Samples per Class:")
    print(df['label'].value_counts())
    
    if 'session_id' not in df.columns:
        # Backwards compatibility if user has old CSV (unlikely since we rewrote collect, but good for robustness)
        # We will synthesize dummy session IDs based on timestamp jumps or just random
        print("Warning: 'session_id' column missing. Using random split.")
        df['session_id'] = 'session_0'
        
    print("\nSessions per Class:")
    print(df.groupby('label')['session_id'].nunique())
    
    # Warnings
    counts = df['label'].value_counts()
    if counts.min() < 300:
        print("\n[WARNING] Some classes have < 300 samples. Collect more data for robustness.")
    if counts.max() / counts.min() > 2.0:
        print("\n[WARNING] Data imbalance detected > 2x. Stratification/Balancing will be used.")

    # 3. Preparation
    # Features start from col 3 (session, timestamp, label, f0, f1...)
    # Actually header is session, timestamp, label, f0...
    feature_cols = [c for c in df.columns if c.startswith('f')]
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['session_id'].values
    
    # 4. Split (Session-based)
    # If we have multiple sessions, we split by session. If only 1 session, fall back to random.
    if len(np.unique(groups)) > 1:
        print(f"\nSplitting by session ({len(np.unique(groups))} total)...")
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups))
    else:
        print("\nOnly 1 session found. Using random time-based split (last 20%).")
        # Time-based split to avoid leakage
        split_pt = int(len(X) * 0.8)
        train_idx = np.arange(split_pt)
        val_idx = np.arange(split_pt, len(X))

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    print(f"Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    # 5. Model Pipeline
    # Using pipeline simplifies saving scaler + model
    
    if model_type == 'rf':
        base_clf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
        # RF doesn't strictly need calibration but helpful
        calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)
    else:
        # Logistic Regression is usually well calibrated, but we ensure it
        base_clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, solver='lbfgs')
        calibrated_clf = CalibratedClassifierCV(base_clf, method='sigmoid', cv=3)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', calibrated_clf)
    ])

    print(f"\nTraining {model_type.upper()} with calibration...")
    # Note: CalibratedClassifierCV with cv=prefit requires fitting base first, 
    # but with cv=int it fits internally on folds. However, we want to use all training data.
    # We will use cv=3 (folds of training data) to calibrate.
    pipeline.fit(X_train, y_train)

    # 6. Evaluation
    print("\n--- Evaluation on Validation Set ---")
    y_pred = pipeline.predict(X_val)
    # Check classes
    classes = pipeline.classes_
    print(f"Classes: {classes}")
    
    print(classification_report(y_val, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred, labels=classes))

    # 7. Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Save pipeline
    joblib.dump(pipeline, MODEL_PATH)
    
    # Save label order for app consistency
    with open(LABEL_ORDER_PATH, 'w') as f:
        json.dump(classes.tolist(), f)
        
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved label order to {LABEL_ORDER_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['logreg', 'rf'], default='logreg', help="Model type")
    args = parser.parse_args()
    
    train_model(args.model)
