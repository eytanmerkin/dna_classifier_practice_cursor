"""
Data analysis for v2.0 dataset with class grouping.

This script analyzes the v2 dataset and compares it to v1.
"""

import csv
import os
from collections import Counter

DATA_DIR_V1 = "DNA_seq_pred_cleaned"
DATA_DIR_V2 = "DNA_seq_pred_cleaned_v2"
OUTPUT_DIR = "analysis_output"
REPORT_FILE = os.path.join(OUTPUT_DIR, "report_v2.md")

NON_CODING_RNA_TYPES = {'ncRNA', 'snoRNA', 'snRNA', 'scRNA', 'tRNA', 'rRNA'}


def load_csv(filepath):
    """Load CSV file."""
    if not os.path.exists(filepath):
        return []
    
    data = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def analyze_dataset():
    """Analyze v2 dataset and compare to v1."""
    
    print("="*60)
    print("Data Analysis v2.0")
    print("="*60)
    
    # Load v1 data
    print("\nLoading v1 data...")
    v1_train = load_csv(os.path.join(DATA_DIR_V1, "train.csv"))
    v1_val = load_csv(os.path.join(DATA_DIR_V1, "validation.csv"))
    v1_test = load_csv(os.path.join(DATA_DIR_V1, "test.csv"))
    v1_all = v1_train + v1_val + v1_test
    
    # Load v2 data
    print("Loading v2 data...")
    v2_train = load_csv(os.path.join(DATA_DIR_V2, "train.csv"))
    v2_val = load_csv(os.path.join(DATA_DIR_V2, "validation.csv"))
    v2_test = load_csv(os.path.join(DATA_DIR_V2, "test.csv"))
    v2_all = v2_train + v2_val + v2_test
    
    # Analyze v1 class distribution
    v1_types = Counter(row.get('GeneType', '') for row in v1_all)
    
    # Analyze v2 class distribution
    v2_types = Counter(row.get('GeneType', '') for row in v2_all)
    
    # Count non-coding RNA types in v1 (before grouping)
    v1_ncrna_count = sum(v1_types.get(gt, 0) for gt in NON_CODING_RNA_TYPES)
    
    # Generate report
    report_lines = [
        "# Data Analysis Report v2.0",
        "",
        "## Overview",
        "",
        "This report compares the v1 (original) and v2 (grouped classes) datasets.",
        "",
        "## Dataset Sizes",
        "",
        "### Version 1 (Original Classes)",
        "",
        f"- Train: {len(v1_train)}",
        f"- Validation: {len(v1_val)}",
        f"- Test: {len(v1_test)}",
        f"- **Total**: {len(v1_all)}",
        "",
        "### Version 2 (Grouped Classes)",
        "",
        f"- Train: {len(v2_train)}",
        f"- Validation: {len(v2_val)}",
        f"- Test: {len(v2_test)}",
        f"- **Total**: {len(v2_all)}",
        "",
        "## Class Distribution Comparison",
        "",
        "### Version 1 - Original Classes (10 classes)",
        "",
        "| Class | Count |",
        "|-------|-------|"
    ]
    
    for gt, count in v1_types.most_common():
        report_lines.append(f"| {gt} | {count} |")
    
    report_lines.extend([
        "",
        f"**Non-coding RNA types (before grouping)**: {v1_ncrna_count}",
        "",
        "### Version 2 - Grouped Classes (5 classes)",
        "",
        "| Class | Count |",
        "|-------|-------|"
    ])
    
    for gt, count in v2_types.most_common():
        report_lines.append(f"| {gt} | {count} |")
    
    report_lines.extend([
        "",
        f"**NON_CODING_RNA (after grouping)**: {v2_types.get('NON_CODING_RNA', 0)}",
        "",
        "## Class Grouping Details",
        "",
        "The following 6 classes were grouped into `NON_CODING_RNA`:",
        "",
        "| Original Class | Count in v1 |",
        "|----------------|-------------|"
    ])
    
    for gt in sorted(NON_CODING_RNA_TYPES):
        count = v1_types.get(gt, 0)
        report_lines.append(f"| {gt} | {count} |")
    
    report_lines.extend([
        "",
        f"**Total grouped**: {v1_ncrna_count} → `NON_CODING_RNA`",
        "",
        "## Key Changes",
        "",
        "1. **Class Reduction**: 10 classes → 5 classes",
        "2. **NON_CODING_RNA**: Combines 6 rare RNA types into one class",
        "3. **Better Balance**: Grouped class has more samples, improving model training",
        "",
        "## Expected Benefits",
        "",
        "1. **Improved Performance on Rare Classes**:",
        "   - Individual RNA types had very few samples (1-169 in test set)",
        "   - Grouped class will have more training examples",
        "",
        "2. **Reduced Overfitting**:",
        "   - Less risk of memorizing tiny classes",
        "   - Better generalization",
        "",
        "3. **Biological Rationale**:",
        "   - All non-coding RNA types share similar functions",
        "   - Grouping is biologically meaningful",
        "",
        "## Training Set Distribution (v2)",
        "",
        "| Class | Count |",
        "|-------|-------|"
    ])
    
    v2_train_types = Counter(row.get('GeneType', '') for row in v2_train)
    for gt, count in v2_train_types.most_common():
        report_lines.append(f"| {gt} | {count} |")
    
    report_lines.append("")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nAnalysis complete. Report saved to: {REPORT_FILE}")
    print("\nSummary:")
    print(f"  v1 classes: {len(v1_types)}")
    print(f"  v2 classes: {len(v2_types)}")
    print(f"  v1 total samples: {len(v1_all)}")
    print(f"  v2 total samples: {len(v2_all)}")


if __name__ == "__main__":
    analyze_dataset()
