import csv
import random
import os
from collections import Counter

# Set random seed for reproducibility
random.seed(42)

INPUT_DIR = "DNA_seq_pred"
OUTPUT_DIR = "DNA_seq_pred_cleaned"

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
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Saved {len(data)} rows to {filepath}")

def clean_dataset():
    print("Loading original data...")
    try:
        train_data = load_csv(os.path.join(INPUT_DIR, "train.csv"))
        test_data = load_csv(os.path.join(INPUT_DIR, "test.csv"))
        val_data = load_csv(os.path.join(INPUT_DIR, "validation.csv"))
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    fieldnames = list(train_data[0].keys())
    # Remove 'GeneGroupMethod' if it exists in fieldnames as it is redundant
    if 'GeneGroupMethod' in fieldnames:
        fieldnames.remove('GeneGroupMethod')

    print(f"Original sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # 1. Remove Leakage (Test sequences that appear in Train)
    # Strategy: Remove the overlapping sequences from TEST set to preserve Training size
    train_sequences = set(row['NucleotideSequence'] for row in train_data)
    
    initial_test_size = len(test_data)
    test_data_cleaned = [row for row in test_data if row['NucleotideSequence'] not in train_sequences]
    
    removed_leakage = initial_test_size - len(test_data_cleaned)
    print(f"Removed {removed_leakage} leaking sequences from Test set.")

    # 2. Remove Duplicates within Train set
    # Strategy: Keep only the first occurrence of a sequence
    seen_sequences = set()
    train_data_deduped = []
    duplicates_removed = 0
    
    for row in train_data:
        seq = row['NucleotideSequence']
        if seq not in seen_sequences:
            seen_sequences.add(seq)
            train_data_deduped.append(row)
        else:
            duplicates_removed += 1
            
    print(f"Removed {duplicates_removed} duplicate sequences from Train set.")

    # 3. Handle Class Imbalance in Train set
    # Strategy: Undersample majority classes to a reasonable cap, Oversample minority classes
    # Analysis showed PSEUDO ~16k, others ~4k, some < 10.
    # Let's target a balanced distribution where possible.
    # Given the small size of some classes, we can't perfectly balance without massive oversampling which is risky.
    # Hybrid approach: 
    # - Cap max samples per class (e.g., 2000)
    # - Upsample small classes to a minimum (e.g., 500) or leave as is if very small?
    # Let's start by just separating them by class.

    class_buckets = {}
    for row in train_data_deduped:
        label = row['GeneType']
        if label not in class_buckets:
            class_buckets[label] = []
        class_buckets[label].append(row)

    print("\nClass distribution in Train (after dedup):")
    for label, rows in class_buckets.items():
        print(f"{label}: {len(rows)}")

    # Balancing Strategy
    TARGET_MAX = 2000  # Undersample strictly dominant classes to this
    TARGET_MIN = 200   # Oversample tiny classes to at least this (if they have > 5 samples)
    
    balanced_train_data = []
    
    for label, rows in class_buckets.items():
        count = len(rows)
        
        if count > TARGET_MAX:
            # Undersample
            selected = random.sample(rows, TARGET_MAX)
            balanced_train_data.extend(selected)
        elif count < TARGET_MIN and count > 10:
             # Oversample (bootstrap)
             # Calculate how many times to repeat
             repeats = TARGET_MIN // count
             remainder = TARGET_MIN % count
             
             extended_rows = rows * repeats + random.sample(rows, remainder)
             balanced_train_data.extend(extended_rows)
        else:
            # Keep as is (Middle sized classes OR extremely small ones where oversampling leads to massive overfitting)
            balanced_train_data.extend(rows)

    random.shuffle(balanced_train_data)
    print(f"\nBalanced Train size: {len(balanced_train_data)}")
    
    # Remove Redundant Column
    for row in balanced_train_data:
        row.pop('GeneGroupMethod', None)
    for row in test_data_cleaned:
        row.pop('GeneGroupMethod', None)
    for row in val_data:
        row.pop('GeneGroupMethod', None)
        # Also clean val data duplicates? Usually better to keep val distinct but representative.
        # We will keep val as is (except removing bad cols) to reflect real world distribution, 
        # OR we should dedup it too. Let's dedup val to be safe.
    
    # Dedup Validation
    val_seen = set()
    val_deduped = []
    for row in val_data:
        seq = row['NucleotideSequence']
        if seq not in val_seen:
            val_seen.add(seq)
            val_deduped.append(row)
    
    print(f"Final sizes - Train: {len(balanced_train_data)}, Val: {len(val_deduped)}, Test: {len(test_data_cleaned)}")

    # Save
    save_csv(balanced_train_data, os.path.join(OUTPUT_DIR, "train.csv"), fieldnames)
    save_csv(test_data_cleaned, os.path.join(OUTPUT_DIR, "test.csv"), fieldnames)
    save_csv(val_deduped, os.path.join(OUTPUT_DIR, "validation.csv"), fieldnames)

if __name__ == "__main__":
    clean_dataset()

