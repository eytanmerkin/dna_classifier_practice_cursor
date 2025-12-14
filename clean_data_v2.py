"""
Clean dataset v2.0 with class grouping support.

This script applies class grouping (non-coding RNA types → NON_CODING_RNA)
before cleaning and balancing the data.
"""

import csv
import random
import os
from collections import Counter

# Set random seed for reproducibility
random.seed(42)

INPUT_DIR = "DNA_seq_pred_v2"  # Use merged dataset
OUTPUT_DIR = "DNA_seq_pred_cleaned_v2"

# Non-coding RNA types to group
NON_CODING_RNA_TYPES = {'ncRNA', 'snoRNA', 'snRNA', 'scRNA', 'tRNA', 'rRNA'}


def map_gene_type(gene_type):
    """Map gene type to grouped version."""
    if gene_type in NON_CODING_RNA_TYPES:
        return 'NON_CODING_RNA'
    return gene_type


def load_csv(filepath):
    data = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean sequence immediately upon loading
            if 'NucleotideSequence' in row:
                row['NucleotideSequence'] = row['NucleotideSequence'].strip('<>')
            data.append(row)
    return data


def save_csv(data, filepath, fieldnames):
    if not data:
        return
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} rows to {filepath}")


def clean_dataset():
    print("="*60)
    print("Cleaning Dataset v2.0 (with Class Grouping)")
    print("="*60)
    
    print("\nLoading data...")
    try:
        train_data = load_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_data = load_csv(os.path.join(INPUT_DIR, "test.csv"))
        val_data = load_csv(os.path.join(INPUT_DIR, "validation.csv"))
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    fieldnames = ['NCBIGeneID', 'Symbol', 'Description', 'GeneType', 'NucleotideSequence']
    
    # Remove GeneGroupMethod if present
    for row in train_data + test_data + val_data:
        row.pop('GeneGroupMethod', None)
        row.pop('', None)  # Remove empty keys

    print(f"Original sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Apply class grouping
    print("\nApplying class grouping...")
    for row in train_data:
        row['GeneType'] = map_gene_type(row.get('GeneType', ''))
    for row in test_data:
        row['GeneType'] = map_gene_type(row.get('GeneType', ''))
    for row in val_data:
        row['GeneType'] = map_gene_type(row.get('GeneType', ''))
    
    # Show grouped distribution
    train_types = Counter(row.get('GeneType', '') for row in train_data)
    print("\nClass distribution after grouping (Train):")
    for label, count in train_types.most_common():
        print(f"  {label}: {count}")

    # 1. Remove Leakage (Test sequences that appear in Train)
    train_sequences = set(row.get('NucleotideSequence', '') for row in train_data)
    
    initial_test_size = len(test_data)
    test_data_cleaned = [row for row in test_data 
                        if row.get('NucleotideSequence', '') not in train_sequences]
    
    removed_leakage = initial_test_size - len(test_data_cleaned)
    print(f"\nRemoved {removed_leakage} leaking sequences from Test set.")

    # 2. Remove Duplicates within Train set
    seen_sequences = set()
    train_data_deduped = []
    duplicates_removed = 0
    
    for row in train_data:
        seq = row.get('NucleotideSequence', '')
        if seq and seq not in seen_sequences:
            seen_sequences.add(seq)
            train_data_deduped.append(row)
        else:
            duplicates_removed += 1
            
    print(f"Removed {duplicates_removed} duplicate sequences from Train set.")

    # 3. Handle Class Imbalance in Train set (with grouped classes)
    class_buckets = {}
    for row in train_data_deduped:
        label = row.get('GeneType', '')
        if label not in class_buckets:
            class_buckets[label] = []
        class_buckets[label].append(row)

    print("\nClass distribution in Train (after dedup, grouped):")
    for label, rows in class_buckets.items():
        print(f"  {label}: {len(rows)}")

    # Balancing Strategy (adjusted for fewer classes)
    TARGET_MAX = 3000  # Higher cap since we have fewer classes now
    TARGET_MIN = 500   # Higher minimum for better representation
    
    balanced_train_data = []
    
    for label, rows in class_buckets.items():
        count = len(rows)
        
        if count > TARGET_MAX:
            # Undersample
            selected = random.sample(rows, TARGET_MAX)
            balanced_train_data.extend(selected)
        elif count < TARGET_MIN and count > 10:
            # Oversample (bootstrap)
            repeats = TARGET_MIN // count
            remainder = TARGET_MIN % count
            
            extended_rows = rows * repeats + random.sample(rows, remainder)
            balanced_train_data.extend(extended_rows)
        else:
            # Keep as is
            balanced_train_data.extend(rows)

    random.shuffle(balanced_train_data)
    print(f"\nBalanced Train size: {len(balanced_train_data)}")
    
    # Final class distribution
    final_types = Counter(row.get('GeneType', '') for row in balanced_train_data)
    print("\nFinal class distribution (balanced):")
    for label, count in final_types.most_common():
        print(f"  {label}: {count}")
    
    # Dedup Validation
    val_seen = set()
    val_deduped = []
    for row in val_data:
        seq = row.get('NucleotideSequence', '')
        if seq and seq not in val_seen:
            val_seen.add(seq)
            val_deduped.append(row)
    
    print(f"\nFinal sizes - Train: {len(balanced_train_data)}, Val: {len(val_deduped)}, Test: {len(test_data_cleaned)}")

    # Save
    save_csv(balanced_train_data, os.path.join(OUTPUT_DIR, "train.csv"), fieldnames)
    save_csv(test_data_cleaned, os.path.join(OUTPUT_DIR, "test.csv"), fieldnames)
    save_csv(val_deduped, os.path.join(OUTPUT_DIR, "validation.csv"), fieldnames)
    
    print("\n" + "="*60)
    print("Cleaning Complete!")
    print("="*60)
    print(f"\nCleaned dataset saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    clean_dataset()
