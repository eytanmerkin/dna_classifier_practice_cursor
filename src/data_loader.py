"""
Data loading and preprocessing utilities for DNA classification.

This module handles loading CSV data, extracting features, and encoding labels.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, Optional

try:
    from .features import batch_extract_features, get_feature_names
except ImportError:
    from features import batch_extract_features, get_feature_names


def map_gene_types_to_grouped(gene_type: str) -> str:
    """
    Map individual gene types to grouped classes.
    
    Groups all non-coding RNA types into a single NON_CODING_RNA class:
    - ncRNA, snoRNA, snRNA, scRNA, tRNA, rRNA → NON_CODING_RNA
    - All other classes remain unchanged
    
    Args:
        gene_type: Original gene type string
        
    Returns:
        Grouped gene type string
    """
    non_coding_rna_types = {'ncRNA', 'snoRNA', 'snRNA', 'scRNA', 'tRNA', 'rRNA'}
    
    if gene_type in non_coding_rna_types:
        return 'NON_CODING_RNA'
    
    return gene_type


def load_data(data_dir: str = "DNA_seq_pred_cleaned") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load train, validation, and test datasets from CSV files.
    
    Args:
        data_dir: Directory containing the CSV files
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "validation.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    print(f"Loading data from {data_dir}/...")
    
    train_df = pd.read_csv(train_path, index_col=0)
    val_df = pd.read_csv(val_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(val_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df


def prepare_features(df: pd.DataFrame, k: int = 4, 
                     show_progress: bool = True) -> np.ndarray:
    """
    Extract k-mer features from sequences in a DataFrame.
    
    Args:
        df: DataFrame with 'NucleotideSequence' column
        k: Length of k-mers
        show_progress: Whether to show progress updates
        
    Returns:
        Feature matrix of shape (n_samples, 4^k)
    """
    sequences = df['NucleotideSequence'].tolist()
    
    if show_progress:
        print(f"Extracting {k}-mer features...")
    
    features = batch_extract_features(sequences, k=k, 
                                       normalize=True, 
                                       show_progress=show_progress)
    return features


def encode_labels(labels: pd.Series, 
                  encoder: Optional[LabelEncoder] = None) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to integers.
    
    Args:
        labels: Series of string labels (e.g., 'PSEUDO', 'ncRNA')
        encoder: Optional pre-fitted LabelEncoder (for test data)
        
    Returns:
        Tuple of (encoded labels array, LabelEncoder)
    """
    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels)
    else:
        encoded = encoder.transform(labels)
    
    return encoded, encoder


def prepare_dataset(data_dir: str = "DNA_seq_pred_cleaned", 
                    k: int = 4,
                    use_grouped_classes: bool = False) -> dict:
    """
    Load and prepare the full dataset for training.
    
    Args:
        data_dir: Directory containing CSV files
        k: K-mer length for feature extraction
        
    Returns:
        Dictionary with X_train, X_val, X_test, y_train, y_val, y_test,
        label_encoder, and feature_names
    """
    # Load data
    train_df, val_df, test_df = load_data(data_dir)
    
    # Extract features
    print("\nExtracting training features...")
    X_train = prepare_features(train_df, k=k)
    
    print("\nExtracting validation features...")
    X_val = prepare_features(val_df, k=k)
    
    print("\nExtracting test features...")
    X_test = prepare_features(test_df, k=k)
    
    # Apply class grouping if requested
    if use_grouped_classes:
        print("\nApplying class grouping...")
        train_df = train_df.copy()
        val_df = val_df.copy()
        test_df = test_df.copy()
        
        train_df['GeneType'] = train_df['GeneType'].apply(map_gene_types_to_grouped)
        val_df['GeneType'] = val_df['GeneType'].apply(map_gene_types_to_grouped)
        test_df['GeneType'] = test_df['GeneType'].apply(map_gene_types_to_grouped)
        
        print("  Grouped non-coding RNA types into NON_CODING_RNA")
    
    # Encode labels
    print("\nEncoding labels...")
    y_train, label_encoder = encode_labels(train_df['GeneType'])
    y_val, _ = encode_labels(val_df['GeneType'], label_encoder)
    y_test, _ = encode_labels(test_df['GeneType'], label_encoder)
    
    # Get feature names
    feature_names = get_feature_names(k)
    
    # Get class names
    class_names = label_encoder.classes_.tolist()
    
    print(f"\nDataset prepared:")
    print(f"  Features: {X_train.shape[1]} ({k}-mers)")
    print(f"  Classes: {len(class_names)} - {class_names}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'class_names': class_names
    }


def get_class_weights(y: np.ndarray) -> dict:
    """
    Calculate class weights for handling imbalanced data.
    
    Args:
        y: Array of encoded labels
        
    Returns:
        Dictionary mapping class index to weight
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    
    return dict(zip(classes, weights))


if __name__ == "__main__":
    # Test data loading
    print("Testing data loader...")
    
    try:
        data = prepare_dataset(k=4)
        
        print(f"\nShapes:")
        print(f"  X_train: {data['X_train'].shape}")
        print(f"  X_val: {data['X_val'].shape}")
        print(f"  X_test: {data['X_test'].shape}")
        
        print(f"\nClass distribution in training:")
        unique, counts = np.unique(data['y_train'], return_counts=True)
        for cls, cnt in zip(unique, counts):
            class_name = data['class_names'][cls]
            print(f"  {class_name}: {cnt}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure to run from the project root directory.")

