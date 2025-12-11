# Data Analysis Report

## 1. Summary of Variables and Labels (3.3.1)

**Total Samples**: 35496
- Train: 22593
- Validation: 4577
- Test: 8326

### Columns:

- ****: 18474 unique values
- **NCBIGeneID**: 23779 unique values
- **Symbol**: 23690 unique values
- **Description**: 22061 unique values
- **GeneType**: 10 unique values
- **GeneGroupMethod**: 1 unique values
- **NucleotideSequence**: 22886 unique values

### Target Variable Analysis (GeneType):
| GeneType | Count |
| --- | --- |
| PSEUDO | 16153 |
| BIOLOGICAL_REGION | 10974 |
| ncRNA | 3907 |
| snoRNA | 1792 |
| PROTEIN_CODING | 809 |
| tRNA | 708 |
| OTHER | 587 |
| rRNA | 357 |
| snRNA | 205 |
| scRNA | 4 |

## 2. Correlations and Redundant Variables (3.3.2)

**Redundant (Constant) Variables**: GeneGroupMethod
**High Cardinality Variables (Likely IDs)**: 

**Sequence Lengths**: Min=2, Max=1000, Mean=361.47

## 3. Data Problems and Fixes (3.3.3)

### Missing Values: None detected.

### Duplicate Sequences: 10954 unique sequences appear multiple times (Total 12610 duplicate entries).
- **CRITICAL**: 7204 sequences found in both Train and Test sets (Data Leakage).
  - **Fix**: Remove these sequences from the test set.

### Invalid Characters: 0 sequences with non-ACGTN characters.

## 4. Summary of Conclusions (3.3.4)

1. Dataset has 35496 samples.
2. 10 classes in GeneType. Top class: ('PSEUDO', 16153)
3. Sequence length range: 2-1000.
4. **Action Required**: Fix data leakage of 7204 samples.