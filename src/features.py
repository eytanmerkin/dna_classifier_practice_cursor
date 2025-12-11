"""
K-mer feature extraction for DNA sequences.

This module provides functions to convert DNA sequences into numerical 
feature vectors using k-mer counting.
"""

import itertools
from collections import Counter
from typing import List, Dict
import numpy as np


def generate_all_kmers(k: int) -> List[str]:
    """
    Generate all possible k-mers of length k.
    
    Args:
        k: Length of k-mers (e.g., 4 for tetramers)
        
    Returns:
        List of all possible k-mers (4^k combinations)
    """
    nucleotides = ['A', 'C', 'G', 'T']
    kmers = [''.join(p) for p in itertools.product(nucleotides, repeat=k)]
    return sorted(kmers)


def count_kmers(sequence: str, k: int) -> Counter:
    """
    Count occurrences of each k-mer in a sequence.
    
    Args:
        sequence: DNA sequence string (A, C, G, T characters)
        k: Length of k-mers to count
        
    Returns:
        Counter with k-mer counts
    """
    sequence = sequence.upper().replace('N', '')  # Remove unknown nucleotides
    kmers = []
    
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        # Only include valid k-mers (no non-ACGT characters)
        if all(c in 'ACGT' for c in kmer):
            kmers.append(kmer)
    
    return Counter(kmers)


def extract_kmer_features(sequence: str, k: int = 4, normalize: bool = True) -> Dict[str, float]:
    """
    Extract k-mer features from a DNA sequence.
    
    Args:
        sequence: DNA sequence string
        k: Length of k-mers (default 4)
        normalize: If True, normalize counts by total k-mers
        
    Returns:
        Dictionary mapping k-mer to count/frequency
    """
    counts = count_kmers(sequence, k)
    all_kmers = generate_all_kmers(k)
    
    # Initialize all k-mers with 0
    features = {kmer: 0.0 for kmer in all_kmers}
    
    # Fill in observed counts
    for kmer, count in counts.items():
        if kmer in features:
            features[kmer] = count
    
    # Normalize if requested
    if normalize:
        total = sum(features.values())
        if total > 0:
            features = {kmer: count / total for kmer, count in features.items()}
    
    return features


def sequence_to_feature_vector(sequence: str, k: int = 4, 
                                kmer_list: List[str] = None,
                                normalize: bool = True) -> np.ndarray:
    """
    Convert a DNA sequence to a numpy feature vector.
    
    Args:
        sequence: DNA sequence string
        k: Length of k-mers
        kmer_list: Ordered list of k-mers (for consistent feature ordering)
        normalize: If True, normalize counts
        
    Returns:
        numpy array of k-mer frequencies
    """
    if kmer_list is None:
        kmer_list = generate_all_kmers(k)
    
    features = extract_kmer_features(sequence, k, normalize)
    return np.array([features.get(kmer, 0.0) for kmer in kmer_list])


def batch_extract_features(sequences: List[str], k: int = 4, 
                           normalize: bool = True,
                           show_progress: bool = True) -> np.ndarray:
    """
    Extract k-mer features from multiple sequences.
    
    Args:
        sequences: List of DNA sequences
        k: Length of k-mers
        normalize: If True, normalize counts
        show_progress: If True, print progress updates
        
    Returns:
        numpy array of shape (n_sequences, 4^k)
    """
    kmer_list = generate_all_kmers(k)
    n_features = len(kmer_list)
    n_sequences = len(sequences)
    
    features_matrix = np.zeros((n_sequences, n_features))
    
    for i, seq in enumerate(sequences):
        if show_progress and (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{n_sequences} sequences...")
        features_matrix[i] = sequence_to_feature_vector(seq, k, kmer_list, normalize)
    
    if show_progress:
        print(f"  Completed: {n_sequences} sequences -> {n_features} features each")
    
    return features_matrix


def get_feature_names(k: int = 4) -> List[str]:
    """
    Get the ordered list of feature names (k-mers).
    
    Args:
        k: Length of k-mers
        
    Returns:
        List of k-mer strings in consistent order
    """
    return generate_all_kmers(k)


if __name__ == "__main__":
    # Test the feature extraction
    test_seq = "ACGTACGTACGTAAAA"
    print(f"Test sequence: {test_seq}")
    print(f"Length: {len(test_seq)}")
    
    # Test k=3
    features_k3 = extract_kmer_features(test_seq, k=3)
    non_zero_k3 = {k: v for k, v in features_k3.items() if v > 0}
    print(f"\nK=3 features (non-zero): {non_zero_k3}")
    
    # Test k=4
    features_k4 = extract_kmer_features(test_seq, k=4)
    non_zero_k4 = {k: v for k, v in features_k4.items() if v > 0}
    print(f"\nK=4 features (non-zero): {non_zero_k4}")
    
    # Test vector conversion
    vec = sequence_to_feature_vector(test_seq, k=4)
    print(f"\nFeature vector shape: {vec.shape}")
    print(f"Sum of features: {vec.sum():.4f}")

