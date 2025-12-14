#!/usr/bin/env python3
"""
DNA Sequence Classifier - Main Entry Point

Usage:
    python src/main.py --train --model rf      # Train Random Forest
    python src/main.py --train --model xgb     # Train XGBoost
    python src/main.py --train --model both    # Train both and pick best
    python src/main.py --evaluate              # Evaluate saved model on test set
    python src/main.py --train --evaluate      # Train and evaluate
"""

import argparse
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import prepare_dataset
from train import (
    train_random_forest, 
    train_xgboost, 
    train_and_compare,
    save_model,
    load_model
)
from evaluate import full_evaluation


def main():
    parser = argparse.ArgumentParser(
        description='DNA Sequence Classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py --train --model rf
  python src/main.py --train --model xgb --tune
  python src/main.py --train --model both
  python src/main.py --evaluate
  python src/main.py --train --evaluate --k 4
        """
    )
    
    parser.add_argument('--train', action='store_true',
                       help='Train a model')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate model on test set')
    parser.add_argument('--model', type=str, default='both',
                       choices=['rf', 'xgb', 'both'],
                       help='Model type: rf (Random Forest), xgb (XGBoost), both')
    parser.add_argument('--k', type=int, default=4,
                       help='K-mer length for feature extraction (default: 4)')
    parser.add_argument('--tune', action='store_true',
                       help='Tune hyperparameters (slower but better results)')
    parser.add_argument('--data-dir', type=str, default='DNA_seq_pred_cleaned',
                       help='Directory containing data files')
    parser.add_argument('--model-path', type=str, default='models/best_model.joblib',
                       help='Path to save/load model')
    parser.add_argument('--version', type=str, default='v1',
                       choices=['v1', 'v2'],
                       help='Version: v1 (original) or v2 (grouped classes)')
    parser.add_argument('--use-new-data', action='store_true',
                       help='Use merged dataset (v2 only, requires DNA_seq_pred_v2)')
    
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        parser.print_help()
        print("\nError: Please specify --train and/or --evaluate")
        sys.exit(1)
    
    print("="*60)
    print("DNA Sequence Classifier")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Version: {args.version}")
    print(f"  K-mer length: {args.k}")
    print(f"  Model type: {args.model}")
    print(f"  Tune hyperparameters: {args.tune}")
    
    # Determine data directory and class grouping
    if args.version == 'v2':
        use_grouped = True
        if args.use_new_data:
            data_dir = 'DNA_seq_pred_cleaned_v2'  # Use v2 cleaned data
        else:
            # Default to v2 cleaned data
            data_dir = 'DNA_seq_pred_cleaned_v2'
        print(f"  Data directory: {data_dir}")
        print(f"  Class grouping: Enabled (NON_CODING_RNA)")
    else:
        use_grouped = False
        data_dir = args.data_dir
        print(f"  Data directory: {data_dir}")
        print(f"  Class grouping: Disabled (original classes)")
    
    # Load and prepare data
    print("\n" + "-"*60)
    data = prepare_dataset(data_dir=data_dir, k=args.k, use_grouped_classes=use_grouped)
    
    model = None
    model_info = None
    
    # Training
    if args.train:
        print("\n" + "-"*60)
        print("TRAINING PHASE")
        print("-"*60)
        
        if args.model == 'rf':
            model, model_info = train_random_forest(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                tune_hyperparams=args.tune
            )
            model_filename = f'best_model_{args.version}.joblib'
            save_model(model, model_info, filename=model_filename)
            
        elif args.model == 'xgb':
            model, model_info = train_xgboost(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                tune_hyperparams=args.tune
            )
            model_filename = f'best_model_{args.version}.joblib'
            save_model(model, model_info, filename=model_filename)
            
        else:  # both
            model, model_info = train_and_compare(
                data['X_train'], data['y_train'],
                data['X_val'], data['y_val'],
                tune_hyperparams=args.tune,
                save_best=True
            )
            # Save with version-specific name
            if model_info:
                model_filename = f'best_model_{args.version}.joblib'
                save_model(model, model_info, filename=model_filename)
    
    # Evaluation
    if args.evaluate:
        print("\n" + "-"*60)
        print("EVALUATION PHASE")
        print("-"*60)
        
        # Load model if not just trained
        if model is None:
            # Try version-specific model path
            if args.model_path == 'models/best_model.joblib':
                model_path = f'models/best_model_{args.version}.joblib'
            else:
                model_path = args.model_path
            print(f"\nLoading model from {model_path}...")
            model, model_info = load_model(model_path)
            
            if model is None:
                print("Error: No trained model found. Run with --train first.")
                sys.exit(1)
        
        # Run full evaluation
        results = full_evaluation(
            model,
            data['X_test'],
            data['y_test'],
            data['class_names'],
            data['feature_names'],
            model_info or {}
        )
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"\n  Test Accuracy:  {results['accuracy']:.4f}")
        print(f"  Test Macro F1:  {results['macro_f1']:.4f}")
        print(f"\nReports generated in: analysis_output/")
        print("  - model_results.md")
        print("  - plots/confusion_matrix.png")
        print("  - plots/feature_importance.png")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()

