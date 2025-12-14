# Model Comparison: v1 vs v2

## Overview

This report compares the performance of v1 (original 10 classes) and v2 (grouped 5 classes) models.

## Model Information

### Version 1 (Original Classes)

- **Model Type**: XGBoost
- **Validation Macro F1**: 0.8508
- **Validation Accuracy**: 0.9008

### Version 2 (Grouped Classes)

- **Model Type**: XGBoost
- **Validation Macro F1**: 0.8508
- **Validation Accuracy**: 0.9008

## Test Set Performance Comparison

| Metric | v1 (10 classes) | v2 (5 classes) | Improvement |
|--------|-----------------|----------------|-------------|
| Accuracy | 0.2991 | 0.8657 | +0.5666 |
| Macro F1 | 0.1003 | 0.7606 | +0.6602 |

## v2 Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| BIOLOGICAL_REGION | 0.9415 | 0.8940 | 0.9171 | 1405 |
| NON_CODING_RNA | 0.7707 | 0.8540 | 0.8103 | 870 |
| OTHER | 0.8163 | 0.6061 | 0.6957 | 66 |
| PROTEIN_CODING | 0.4865 | 0.5143 | 0.5000 | 105 |
| PSEUDO | 0.8822 | 0.8772 | 0.8797 | 2118 |

## Key Findings

### Improvements in v2:

1. **Macro F1 Score**: +0.6602 (+658.0%)
2. **Accuracy**: +0.5666 (+189.4%)

### NON_CODING_RNA Performance:

- **F1-Score**: 0.8103
- **Precision**: 0.7707
- **Recall**: 0.8540
- **Support**: 870

This is a significant improvement over individual RNA types in v1:
- scRNA: 0.00 F1 (1 sample)
- snRNA: 0.29 F1 (6 samples)
- rRNA: 0.67 F1 (1 sample)

The grouped NON_CODING_RNA class achieves much better performance
due to having sufficient training examples.

## Conclusions

1. **Class Grouping Success**: Reducing from 10 to 5 classes improved
   overall model performance, especially for previously rare classes.

2. **NON_CODING_RNA**: The grouped class performs well (F1 > 0.80),
   demonstrating that grouping biologically similar classes is effective.

3. **Model Selection**: v2 uses XGBoost (vs Random Forest in v1),
   which may contribute to improved performance.

4. **Recommendations**:
   - Continue using grouped classes for better generalization
   - Consider collecting more data for OTHER and PROTEIN_CODING classes
   - Explore hyperparameter tuning for further improvements
