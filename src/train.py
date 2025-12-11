"""
Model training for DNA sequence classification.

This module provides functions to train Random Forest and XGBoost classifiers.
"""

import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, Any, Tuple, Optional

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Only Random Forest will be available.")


def train_random_forest(X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray,
                        tune_hyperparams: bool = True,
                        n_jobs: int = -1) -> Tuple[RandomForestClassifier, Dict]:
    """
    Train a Random Forest classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        tune_hyperparams: Whether to perform grid search
        n_jobs: Number of parallel jobs (-1 for all cores)
        
    Returns:
        Tuple of (trained model, training info dict)
    """
    print("\n" + "="*50)
    print("Training Random Forest Classifier")
    print("="*50)
    
    if tune_hyperparams:
        print("\nPerforming hyperparameter search...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, 30, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        base_rf = RandomForestClassifier(
            class_weight='balanced',
            random_state=42,
            n_jobs=n_jobs
        )
        
        grid_search = GridSearchCV(
            base_rf, 
            param_grid, 
            cv=3, 
            scoring='f1_macro',
            n_jobs=1,  # Already parallelized internally
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\nBest parameters: {best_params}")
    else:
        print("\nTraining with default parameters...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=n_jobs
        )
        model.fit(X_train, y_train)
        best_params = model.get_params()
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"\nValidation Results:")
    print(f"  Macro F1: {val_f1:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    
    info = {
        'model_type': 'RandomForest',
        'best_params': best_params,
        'val_f1_macro': val_f1,
        'val_accuracy': val_acc
    }
    
    return model, info


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  tune_hyperparams: bool = True,
                  n_jobs: int = -1) -> Tuple[Any, Dict]:
    """
    Train an XGBoost classifier with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        tune_hyperparams: Whether to perform grid search
        n_jobs: Number of parallel jobs
        
    Returns:
        Tuple of (trained model, training info dict)
    """
    if not HAS_XGBOOST:
        raise ImportError("XGBoost is not installed. Run: pip install xgboost")
    
    print("\n" + "="*50)
    print("Training XGBoost Classifier")
    print("="*50)
    
    n_classes = len(np.unique(y_train))
    
    if tune_hyperparams:
        print("\nPerforming hyperparameter search...")
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [6, 10],
            'learning_rate': [0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        
        base_xgb = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=n_classes,
            random_state=42,
            n_jobs=n_jobs,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        
        grid_search = GridSearchCV(
            base_xgb,
            param_grid,
            cv=3,
            scoring='f1_macro',
            n_jobs=1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        print(f"\nBest parameters: {best_params}")
    else:
        print("\nTraining with default parameters...")
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=10,
            learning_rate=0.1,
            subsample=0.8,
            objective='multi:softmax',
            num_class=n_classes,
            random_state=42,
            n_jobs=n_jobs,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
        model.fit(X_train, y_train)
        best_params = model.get_params()
    
    # Evaluate on validation set
    val_preds = model.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average='macro')
    val_acc = accuracy_score(y_val, val_preds)
    
    print(f"\nValidation Results:")
    print(f"  Macro F1: {val_f1:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    
    info = {
        'model_type': 'XGBoost',
        'best_params': best_params,
        'val_f1_macro': val_f1,
        'val_accuracy': val_acc
    }
    
    return model, info


def save_model(model: Any, info: Dict, output_dir: str = "models",
               filename: str = "best_model.joblib") -> str:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained model object
        info: Dictionary with training info
        output_dir: Directory to save model
        filename: Model filename
        
    Returns:
        Path to saved model
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model_path = os.path.join(output_dir, filename)
    info_path = os.path.join(output_dir, filename.replace('.joblib', '_info.joblib'))
    
    joblib.dump(model, model_path)
    joblib.dump(info, info_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Info saved to: {info_path}")
    
    return model_path


def load_model(model_path: str = "models/best_model.joblib") -> Tuple[Any, Optional[Dict]]:
    """
    Load a trained model from disk.
    
    Args:
        model_path: Path to saved model
        
    Returns:
        Tuple of (model, info dict)
    """
    model = joblib.load(model_path)
    
    info_path = model_path.replace('.joblib', '_info.joblib')
    info = None
    if os.path.exists(info_path):
        info = joblib.load(info_path)
    
    return model, info


def train_and_compare(X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray,
                      tune_hyperparams: bool = True,
                      save_best: bool = True) -> Tuple[Any, Dict]:
    """
    Train both Random Forest and XGBoost, compare, and return best.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        tune_hyperparams: Whether to tune hyperparameters
        save_best: Whether to save the best model
        
    Returns:
        Tuple of (best model, info dict)
    """
    results = []
    
    # Train Random Forest
    rf_model, rf_info = train_random_forest(
        X_train, y_train, X_val, y_val, 
        tune_hyperparams=tune_hyperparams
    )
    results.append(('RandomForest', rf_model, rf_info))
    
    # Train XGBoost if available
    if HAS_XGBOOST:
        xgb_model, xgb_info = train_xgboost(
            X_train, y_train, X_val, y_val,
            tune_hyperparams=tune_hyperparams
        )
        results.append(('XGBoost', xgb_model, xgb_info))
    
    # Compare results
    print("\n" + "="*50)
    print("Model Comparison")
    print("="*50)
    
    best_model = None
    best_info = None
    best_f1 = -1
    
    for name, model, info in results:
        f1 = info['val_f1_macro']
        acc = info['val_accuracy']
        print(f"\n{name}:")
        print(f"  Validation Macro F1: {f1:.4f}")
        print(f"  Validation Accuracy: {acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_info = info
    
    print(f"\n*** Best Model: {best_info['model_type']} (F1={best_f1:.4f}) ***")
    
    if save_best:
        save_model(best_model, best_info)
    
    return best_model, best_info


if __name__ == "__main__":
    try:
        from data_loader import prepare_dataset
    except ImportError:
        from .data_loader import prepare_dataset
    
    # Load data
    data = prepare_dataset(k=4)
    
    # Train and compare models
    best_model, best_info = train_and_compare(
        data['X_train'], data['y_train'],
        data['X_val'], data['y_val'],
        tune_hyperparams=False,  # Set to True for full tuning
        save_best=True
    )

