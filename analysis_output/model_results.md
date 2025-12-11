# DNA Classifier - Model Results

**Generated**: 2025-12-11 14:54:26

## Model Information

- **Model Type**: RandomForest
- **Validation Macro F1**: 0.8877
- **Validation Accuracy**: 0.8892

### Best Hyperparameters

```
bootstrap: True
ccp_alpha: 0.0
class_weight: balanced
criterion: gini
max_depth: 30
max_features: sqrt
max_leaf_nodes: None
max_samples: None
min_impurity_decrease: 0.0
min_samples_leaf: 1
min_samples_split: 2
min_weight_fraction_leaf: 0.0
monotonic_cst: None
n_estimators: 200
n_jobs: -1
oob_score: False
random_state: 42
verbose: 0
warm_start: False
```

## Test Set Results

- **Test Accuracy**: 0.8355
- **Test Macro F1**: 0.6168
- **Test Weighted F1**: 0.8325

## Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| BIOLOGICAL_REGION | 0.8667 | 0.9256 | 0.8951 | 309 |
| OTHER | 0.6000 | 0.4286 | 0.5000 | 7 |
| PROTEIN_CODING | 0.6364 | 0.5250 | 0.5753 | 40 |
| PSEUDO | 0.8702 | 0.8182 | 0.8434 | 418 |
| ncRNA | 0.7865 | 0.8284 | 0.8069 | 169 |
| rRNA | 0.5000 | 1.0000 | 0.6667 | 1 |
| scRNA | 0.0000 | 0.0000 | 0.0000 | 1 |
| snRNA | 1.0000 | 0.1667 | 0.2857 | 6 |
| snoRNA | 0.6852 | 0.8810 | 0.7708 | 42 |
| tRNA | 1.0000 | 0.7000 | 0.8235 | 10 |

## Confusion Matrix

![Confusion Matrix](plots/confusion_matrix.png)

## Feature Importance

![Feature Importance](plots/feature_importance.png)

## Classification Report

```
                   precision    recall  f1-score   support

BIOLOGICAL_REGION       0.87      0.93      0.90       309
            OTHER       0.60      0.43      0.50         7
   PROTEIN_CODING       0.64      0.53      0.58        40
           PSEUDO       0.87      0.82      0.84       418
            ncRNA       0.79      0.83      0.81       169
             rRNA       0.50      1.00      0.67         1
            scRNA       0.00      0.00      0.00         1
            snRNA       1.00      0.17      0.29         6
           snoRNA       0.69      0.88      0.77        42
             tRNA       1.00      0.70      0.82        10

         accuracy                           0.84      1003
        macro avg       0.69      0.63      0.62      1003
     weighted avg       0.84      0.84      0.83      1003

```
