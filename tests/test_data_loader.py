"""
Unit tests for data loading and preprocessing (src/data_loader.py).
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data_loader import (
    map_gene_types_to_grouped,
    encode_labels,
    get_class_weights
)


class TestMapGeneTypesToGrouped:
    """Tests for the class grouping function."""
    
    def test_ncrna_grouped(self):
        """ncRNA should be grouped to NON_CODING_RNA."""
        assert map_gene_types_to_grouped('ncRNA') == 'NON_CODING_RNA'
    
    def test_snorna_grouped(self):
        """snoRNA should be grouped to NON_CODING_RNA."""
        assert map_gene_types_to_grouped('snoRNA') == 'NON_CODING_RNA'
    
    def test_snrna_grouped(self):
        """snRNA should be grouped to NON_CODING_RNA."""
        assert map_gene_types_to_grouped('snRNA') == 'NON_CODING_RNA'
    
    def test_scrna_grouped(self):
        """scRNA should be grouped to NON_CODING_RNA."""
        assert map_gene_types_to_grouped('scRNA') == 'NON_CODING_RNA'
    
    def test_trna_grouped(self):
        """tRNA should be grouped to NON_CODING_RNA."""
        assert map_gene_types_to_grouped('tRNA') == 'NON_CODING_RNA'
    
    def test_rrna_grouped(self):
        """rRNA should be grouped to NON_CODING_RNA."""
        assert map_gene_types_to_grouped('rRNA') == 'NON_CODING_RNA'
    
    def test_pseudo_unchanged(self):
        """PSEUDO should remain unchanged."""
        assert map_gene_types_to_grouped('PSEUDO') == 'PSEUDO'
    
    def test_biological_region_unchanged(self):
        """BIOLOGICAL_REGION should remain unchanged."""
        assert map_gene_types_to_grouped('BIOLOGICAL_REGION') == 'BIOLOGICAL_REGION'
    
    def test_protein_coding_unchanged(self):
        """PROTEIN_CODING should remain unchanged."""
        assert map_gene_types_to_grouped('PROTEIN_CODING') == 'PROTEIN_CODING'
    
    def test_other_unchanged(self):
        """OTHER should remain unchanged."""
        assert map_gene_types_to_grouped('OTHER') == 'OTHER'
    
    def test_all_rna_types_grouped(self):
        """All 6 RNA types should be grouped correctly."""
        rna_types = ['ncRNA', 'snoRNA', 'snRNA', 'scRNA', 'tRNA', 'rRNA']
        for rna_type in rna_types:
            assert map_gene_types_to_grouped(rna_type) == 'NON_CODING_RNA'


class TestEncodeLabels:
    """Tests for label encoding function."""
    
    def test_basic_encoding(self):
        """Basic encoding should work correctly."""
        labels = pd.Series(['A', 'B', 'C', 'A', 'B'])
        encoded, encoder = encode_labels(labels)
        
        assert len(encoded) == 5
        assert len(encoder.classes_) == 3
    
    def test_encoding_is_consistent(self):
        """Same label should always get same encoding."""
        labels = pd.Series(['PSEUDO', 'ncRNA', 'PSEUDO', 'OTHER'])
        encoded, encoder = encode_labels(labels)
        
        # All 'PSEUDO' should have same encoding
        pseudo_indices = [i for i, l in enumerate(labels) if l == 'PSEUDO']
        assert encoded[pseudo_indices[0]] == encoded[pseudo_indices[1]]
    
    def test_encoder_reuse(self):
        """Should be able to reuse encoder for new data."""
        train_labels = pd.Series(['A', 'B', 'C'])
        test_labels = pd.Series(['B', 'C', 'A'])
        
        _, encoder = encode_labels(train_labels)
        encoded_test, _ = encode_labels(test_labels, encoder=encoder)
        
        # B should get same encoding in both
        assert encoded_test[0] == encoder.transform(['B'])[0]


class TestGetClassWeights:
    """Tests for class weight calculation."""
    
    def test_balanced_weights(self):
        """Balanced dataset should have equal weights."""
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        weights = get_class_weights(y)
        
        # All weights should be approximately 1
        for w in weights.values():
            assert abs(w - 1.0) < 0.1
    
    def test_imbalanced_weights(self):
        """Minority class should have higher weight."""
        y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 8 vs 2
        weights = get_class_weights(y)
        
        # Class 1 (minority) should have higher weight than class 0
        assert weights[1] > weights[0]
    
    def test_returns_dict(self):
        """Should return a dictionary."""
        y = np.array([0, 1, 2, 0, 1, 2])
        weights = get_class_weights(y)
        
        assert isinstance(weights, dict)
        assert 0 in weights
        assert 1 in weights
        assert 2 in weights


class TestDataFrameGrouping:
    """Integration tests for grouping applied to DataFrames."""
    
    def test_dataframe_grouping(self, sample_dataframe, expected_grouped_types):
        """Test applying grouping to a DataFrame."""
        df = sample_dataframe.copy()
        df['GeneType'] = df['GeneType'].apply(map_gene_types_to_grouped)
        
        assert df['GeneType'].tolist() == expected_grouped_types
    
    def test_grouped_reduces_classes(self, sample_dataframe):
        """Grouping should reduce the number of unique classes."""
        original_classes = sample_dataframe['GeneType'].nunique()
        
        df = sample_dataframe.copy()
        df['GeneType'] = df['GeneType'].apply(map_gene_types_to_grouped)
        grouped_classes = df['GeneType'].nunique()
        
        # We started with 5 classes (PSEUDO, ncRNA, snoRNA, BIOLOGICAL_REGION, tRNA)
        # After grouping: PSEUDO, NON_CODING_RNA, BIOLOGICAL_REGION
        assert grouped_classes < original_classes
        assert grouped_classes == 3
