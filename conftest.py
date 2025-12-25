"""
Pytest fixtures for DNA classifier tests.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


@pytest.fixture
def sample_sequences():
    """Sample DNA sequences for testing."""
    return [
        "ACGTACGTACGT",
        "AAAACCCCGGGGTTTT",
        "ATATATAT",
        "GCGCGCGC",
        "ACGT"
    ]


@pytest.fixture
def sample_sequence_short():
    """A short DNA sequence for testing."""
    return "ACGT"


@pytest.fixture
def sample_sequence_with_n():
    """DNA sequence with unknown nucleotide N."""
    return "ACGTNACGT"


@pytest.fixture
def sample_dataframe():
    """Sample DataFrame for testing data loading functions."""
    return pd.DataFrame({
        'GeneType': ['PSEUDO', 'ncRNA', 'snoRNA', 'BIOLOGICAL_REGION', 'tRNA'],
        'NucleotideSequence': [
            'ACGTACGTACGT',
            'AAAACCCCGGGGTTTT',
            'ATATATAT',
            'GCGCGCGC',
            'ACGTACGT'
        ]
    })


@pytest.fixture
def expected_grouped_types():
    """Expected output after class grouping."""
    return ['PSEUDO', 'NON_CODING_RNA', 'NON_CODING_RNA', 'BIOLOGICAL_REGION', 'NON_CODING_RNA']
