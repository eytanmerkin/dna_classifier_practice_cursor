"""
Unit tests for k-mer feature extraction (src/features.py).
"""
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from features import (
    generate_all_kmers,
    count_kmers,
    extract_kmer_features,
    sequence_to_feature_vector,
    batch_extract_features,
    get_feature_names
)


class TestGenerateAllKmers:
    """Tests for generate_all_kmers function."""
    
    def test_k1_generates_4_kmers(self):
        """1-mers should generate 4 nucleotides."""
        kmers = generate_all_kmers(k=1)
        assert len(kmers) == 4
        assert set(kmers) == {'A', 'C', 'G', 'T'}
    
    def test_k2_generates_16_kmers(self):
        """2-mers should generate 4^2 = 16 combinations."""
        kmers = generate_all_kmers(k=2)
        assert len(kmers) == 16
        assert 'AA' in kmers
        assert 'TT' in kmers
    
    def test_k3_generates_64_kmers(self):
        """3-mers should generate 4^3 = 64 combinations."""
        kmers = generate_all_kmers(k=3)
        assert len(kmers) == 64
    
    def test_k4_generates_256_kmers(self):
        """4-mers should generate 4^4 = 256 combinations."""
        kmers = generate_all_kmers(k=4)
        assert len(kmers) == 256
    
    def test_kmers_are_sorted(self):
        """K-mers should be returned in sorted order."""
        kmers = generate_all_kmers(k=2)
        assert kmers == sorted(kmers)


class TestCountKmers:
    """Tests for count_kmers function."""
    
    def test_simple_sequence(self):
        """Test counting k-mers in a simple sequence."""
        sequence = "AAAA"
        counts = count_kmers(sequence, k=2)
        assert counts['AA'] == 3  # AAA has 3 overlapping 'AA'
    
    def test_handles_lowercase(self):
        """Function should handle lowercase input."""
        sequence = "acgt"
        counts = count_kmers(sequence, k=2)
        assert counts['AC'] == 1
        assert counts['CG'] == 1
        assert counts['GT'] == 1
    
    def test_removes_n_nucleotides(self):
        """N nucleotides should be skipped."""
        sequence = "ACNGT"
        counts = count_kmers(sequence, k=2)
        # With N removed, sequence is effectively "ACGT" but split
        # AC is before N, GT is after N
        assert counts['AC'] == 1
        assert counts['GT'] == 1
        # NG and CN should not exist
        assert 'CN' not in counts
        assert 'NG' not in counts
    
    def test_empty_sequence(self):
        """Empty sequence should return empty counter."""
        sequence = ""
        counts = count_kmers(sequence, k=2)
        assert len(counts) == 0


class TestExtractKmerFeatures:
    """Tests for extract_kmer_features function."""
    
    def test_returns_all_possible_kmers(self):
        """Should return features for all possible k-mers."""
        sequence = "ACGT"
        features = extract_kmer_features(sequence, k=2)
        assert len(features) == 16  # 4^2
    
    def test_normalization(self):
        """Normalized features should sum to 1."""
        sequence = "ACGTACGTACGT"
        features = extract_kmer_features(sequence, k=2, normalize=True)
        total = sum(features.values())
        assert abs(total - 1.0) < 0.0001
    
    def test_no_normalization(self):
        """Without normalization, should return raw counts."""
        sequence = "AAAA"
        features = extract_kmer_features(sequence, k=2, normalize=False)
        assert features['AA'] == 3


class TestSequenceToFeatureVector:
    """Tests for sequence_to_feature_vector function."""
    
    def test_returns_numpy_array(self):
        """Should return a numpy array."""
        sequence = "ACGTACGT"
        vector = sequence_to_feature_vector(sequence, k=2)
        assert isinstance(vector, np.ndarray)
    
    def test_correct_shape_k2(self):
        """Shape should be (16,) for k=2."""
        sequence = "ACGTACGT"
        vector = sequence_to_feature_vector(sequence, k=2)
        assert vector.shape == (16,)
    
    def test_correct_shape_k4(self):
        """Shape should be (256,) for k=4."""
        sequence = "ACGTACGTACGTACGT"
        vector = sequence_to_feature_vector(sequence, k=4)
        assert vector.shape == (256,)
    
    def test_normalized_sum_to_one(self):
        """Normalized vector should sum to approximately 1."""
        sequence = "ACGTACGTACGT"
        vector = sequence_to_feature_vector(sequence, k=2, normalize=True)
        assert abs(vector.sum() - 1.0) < 0.0001


class TestBatchExtractFeatures:
    """Tests for batch_extract_features function."""
    
    def test_batch_shape(self, sample_sequences):
        """Batch extraction should return correct shape."""
        features = batch_extract_features(sample_sequences, k=2, show_progress=False)
        assert features.shape == (5, 16)  # 5 sequences, 16 2-mers
    
    def test_batch_features_match_individual(self, sample_sequences):
        """Batch features should match individual extraction."""
        batch_features = batch_extract_features(sample_sequences, k=2, show_progress=False)
        
        for i, seq in enumerate(sample_sequences):
            individual = sequence_to_feature_vector(seq, k=2)
            np.testing.assert_array_almost_equal(batch_features[i], individual)


class TestGetFeatureNames:
    """Tests for get_feature_names function."""
    
    def test_returns_correct_count(self):
        """Should return 4^k feature names."""
        names = get_feature_names(k=4)
        assert len(names) == 256
    
    def test_names_match_generate_all_kmers(self):
        """Feature names should match generate_all_kmers output."""
        names = get_feature_names(k=3)
        kmers = generate_all_kmers(k=3)
        assert names == kmers
