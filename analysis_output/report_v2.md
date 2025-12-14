# Data Analysis Report v2.0

## Overview

This report compares the v1 (original) and v2 (grouped classes) datasets.

## Dataset Sizes

### Version 1 (Original Classes)

- Train: 8689
- Validation: 4577
- Test: 1003
- **Total**: 14269

### Version 2 (Grouped Classes)

- Train: 10000
- Validation: 4295
- Test: 4564
- **Total**: 18859

## Class Distribution Comparison

### Version 1 - Original Classes (10 classes)

| Class | Count |
|-------|-------|
| PSEUDO | 4551 |
| BIOLOGICAL_REGION | 3707 |
| ncRNA | 2685 |
| snoRNA | 1404 |
| PROTEIN_CODING | 656 |
| OTHER | 456 |
| tRNA | 369 |
| snRNA | 228 |
| rRNA | 209 |
| scRNA | 4 |

**Non-coding RNA types (before grouping)**: 4899

### Version 2 - Grouped Classes (5 classes)

| Class | Count |
|-------|-------|
| PSEUDO | 7095 |
| BIOLOGICAL_REGION | 5765 |
| NON_CODING_RNA | 4666 |
| PROTEIN_CODING | 693 |
| OTHER | 640 |

**NON_CODING_RNA (after grouping)**: 4666

## Class Grouping Details

The following 6 classes were grouped into `NON_CODING_RNA`:

| Original Class | Count in v1 |
|----------------|-------------|
| ncRNA | 2685 |
| rRNA | 209 |
| scRNA | 4 |
| snRNA | 228 |
| snoRNA | 1404 |
| tRNA | 369 |

**Total grouped**: 4899 → `NON_CODING_RNA`

## Key Changes

1. **Class Reduction**: 10 classes → 5 classes
2. **NON_CODING_RNA**: Combines 6 rare RNA types into one class
3. **Better Balance**: Grouped class has more samples, improving model training

## Expected Benefits

1. **Improved Performance on Rare Classes**:
   - Individual RNA types had very few samples (1-169 in test set)
   - Grouped class will have more training examples

2. **Reduced Overfitting**:
   - Less risk of memorizing tiny classes
   - Better generalization

3. **Biological Rationale**:
   - All non-coding RNA types share similar functions
   - Grouping is biologically meaningful

## Training Set Distribution (v2)

| Class | Count |
|-------|-------|
| NON_CODING_RNA | 3000 |
| BIOLOGICAL_REGION | 3000 |
| PSEUDO | 3000 |
| PROTEIN_CODING | 500 |
| OTHER | 500 |
