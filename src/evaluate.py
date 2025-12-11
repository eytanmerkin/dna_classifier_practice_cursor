"""
Model evaluation for DNA sequence classification.

This module provides functions to evaluate trained models and generate reports.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    f1_score, 
    accuracy_score,
    precision_recall_fscore_support
)
from typing import Dict, Any, List, Optional
from datetime import datetime


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: List[str]) -> Dict[str, Any]:
    """
    Evaluate a model on test data.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True test labels
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "="*50)
    print("Model Evaluation on Test Set")
    print("="*50)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Overall metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1-Score: {macro_f1:.4f}")
    print(f"  Weighted F1-Score: {weighted_f1:.4f}")
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    
    print(f"\nPer-Class Metrics:")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    
    per_class = {}
    for i, name in enumerate(class_names):
        if i < len(precision):
            print(f"{name:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f} {support[i]:>10}")
            per_class[name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class': per_class,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_true': y_test
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str],
                         output_path: str = "analysis_output/plots/confusion_matrix.png",
                         normalize: bool = True) -> str:
    """
    Plot and save a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names
        output_path: Path to save the plot
        normalize: Whether to normalize the matrix
        
    Returns:
        Path to saved plot
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if normalize:
        cm_display = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_display = np.nan_to_num(cm_display)  # Handle division by zero
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_display = cm
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_display, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"\nConfusion matrix saved to: {output_path}")
    return output_path


def plot_feature_importance(model: Any, feature_names: List[str],
                           output_path: str = "analysis_output/plots/feature_importance.png",
                           top_n: int = 30) -> Optional[str]:
    """
    Plot feature importance for tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names (k-mers)
        output_path: Path to save the plot
        top_n: Number of top features to show
        
    Returns:
        Path to saved plot or None if not applicable
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model doesn't have feature importances.")
        return None
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Most Important K-mers', fontsize=14)
    plt.bar(range(top_n), importances[indices], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('K-mer', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Feature importance plot saved to: {output_path}")
    return output_path


def generate_report(eval_results: Dict, model_info: Dict, 
                   output_path: str = "analysis_output/model_results.md") -> str:
    """
    Generate a markdown report with evaluation results.
    
    Args:
        eval_results: Results from evaluate_model()
        model_info: Info dict from training
        output_path: Path to save the report
        
    Returns:
        Path to saved report
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    lines = [
        "# DNA Classifier - Model Results",
        "",
        f"**Generated**: {timestamp}",
        "",
        "## Model Information",
        "",
        f"- **Model Type**: {model_info.get('model_type', 'Unknown')}",
        f"- **Validation Macro F1**: {model_info.get('val_f1_macro', 0):.4f}",
        f"- **Validation Accuracy**: {model_info.get('val_accuracy', 0):.4f}",
        "",
        "### Best Hyperparameters",
        "",
        "```",
    ]
    
    best_params = model_info.get('best_params', {})
    for key, value in best_params.items():
        if not key.startswith('_'):
            lines.append(f"{key}: {value}")
    
    lines.extend([
        "```",
        "",
        "## Test Set Results",
        "",
        f"- **Test Accuracy**: {eval_results['accuracy']:.4f}",
        f"- **Test Macro F1**: {eval_results['macro_f1']:.4f}",
        f"- **Test Weighted F1**: {eval_results['weighted_f1']:.4f}",
        "",
        "## Per-Class Performance",
        "",
        "| Class | Precision | Recall | F1-Score | Support |",
        "|-------|-----------|--------|----------|---------|"
    ])
    
    for class_name, metrics in eval_results['per_class'].items():
        lines.append(
            f"| {class_name} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | "
            f"{int(metrics['support'])} |"
        )
    
    lines.extend([
        "",
        "## Confusion Matrix",
        "",
        "![Confusion Matrix](plots/confusion_matrix.png)",
        "",
        "## Feature Importance",
        "",
        "![Feature Importance](plots/feature_importance.png)",
        "",
        "## Classification Report",
        "",
        "```",
        eval_results['classification_report'],
        "```",
        ""
    ])
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"\nReport saved to: {output_path}")
    return output_path


def full_evaluation(model: Any, X_test: np.ndarray, y_test: np.ndarray,
                   class_names: List[str], feature_names: List[str],
                   model_info: Dict) -> Dict:
    """
    Run full evaluation pipeline: metrics, plots, and report.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: True test labels
        class_names: List of class names
        feature_names: List of feature names
        model_info: Training info dict
        
    Returns:
        Evaluation results dictionary
    """
    # Evaluate
    eval_results = evaluate_model(model, X_test, y_test, class_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        eval_results['confusion_matrix'], 
        class_names
    )
    
    # Plot feature importance
    plot_feature_importance(model, feature_names)
    
    # Generate report
    generate_report(eval_results, model_info)
    
    return eval_results


if __name__ == "__main__":
    try:
        from data_loader import prepare_dataset
        from train import load_model
    except ImportError:
        from .data_loader import prepare_dataset
        from .train import load_model
    
    # Load data
    data = prepare_dataset(k=4)
    
    # Load model
    model, info = load_model()
    
    if model is not None:
        # Run evaluation
        results = full_evaluation(
            model,
            data['X_test'],
            data['y_test'],
            data['class_names'],
            data['feature_names'],
            info or {}
        )
    else:
        print("No model found. Train a model first.")

