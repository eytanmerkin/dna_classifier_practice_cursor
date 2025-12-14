"""
Validate and clean newly collected data from NCBI.

This script validates that new data matches the format and quality
standards of the existing dataset.
"""

import csv
import os
from collections import Counter

INPUT_DIR = "DNA_seq_pred_additional"
OUTPUT_DIR = "DNA_seq_pred_additional"
ORIGINAL_DIR = "DNA_seq_pred"
REPORT_FILE = "analysis_output/new_data_stats.md"

VALID_GENE_TYPES = {
    'PSEUDO', 'BIOLOGICAL_REGION', 'ncRNA', 'snoRNA', 'PROTEIN_CODING',
    'tRNA', 'OTHER', 'rRNA', 'snRNA', 'scRNA'
}

VALID_NUCLEOTIDES = set('ACGTN')


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


def validate_sequence(seq):
    """
    Validate sequence format.
    
    Returns:
        (is_valid, cleaned_sequence)
    """
    if not seq:
        return False, ""
    
    # Remove angle brackets if present
    seq = seq.strip('<>').strip()
    
    if not seq:
        return False, ""
    
    # Check length
    if len(seq) < 2 or len(seq) > 1000:
        return False, seq
    
    # Check for valid nucleotides
    seq_upper = seq.upper()
    if not set(seq_upper).issubset(VALID_NUCLEOTIDES):
        return False, seq
    
    return True, seq_upper


def check_duplicates(new_data, original_data):
    """Check for duplicate sequences against original data."""
    original_seqs = set()
    for row in original_data:
        seq = row.get('NucleotideSequence', '').strip('<>')
        if seq:
            original_seqs.add(seq.upper())
    
    duplicates = []
    unique_new = []
    
    for row in new_data:
        seq = row.get('NucleotideSequence', '').strip('<>')
        if seq and seq.upper() in original_seqs:
            duplicates.append(row)
        else:
            unique_new.append(row)
    
    return duplicates, unique_new


def validate_data():
    """Main validation function."""
    
    print("="*60)
    print("Validating New Data")
    print("="*60)
    
    # Load new data
    new_file = os.path.join(INPUT_DIR, "additional_genes.csv")
    new_data = load_csv(new_file)
    
    if not new_data:
        print(f"\nNo data found in {new_file}")
        print("This is expected if NCBI fetching hasn't been run yet.")
        print("The validation script is ready for when data is collected.")
        
        # Generate empty report
        report_lines = [
            "# New Data Validation Report",
            "",
            "## Status",
            "",
            "No new data found to validate.",
            "",
            "## Validation Criteria",
            "",
            "When data is available, this script will validate:",
            "",
            "1. **Sequence Format**:",
            "   - ACGTN characters only",
            "   - Length: 2-1000 nucleotides",
            "   - No invalid characters",
            "",
            "2. **Column Structure**:",
            "   - Required columns: NCBIGeneID, Symbol, Description, GeneType, NucleotideSequence",
            "   - Consistent with original data format",
            "",
            "3. **Gene Types**:",
            "   - Must be one of the known types",
            "   - Valid values: PSEUDO, BIOLOGICAL_REGION, ncRNA, snoRNA, etc.",
            "",
            "4. **Duplicates**:",
            "   - Check against original dataset",
            "   - Remove duplicates to prevent data leakage",
            "",
            "5. **Data Quality**:",
            "   - Complete records (no missing critical fields)",
            "   - Valid gene IDs",
            ""
        ]
        
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
        with open(REPORT_FILE, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"\nValidation template report saved to: {REPORT_FILE}")
        return
    
    print(f"\nLoaded {len(new_data)} records from new data")
    
    # Load original data for duplicate checking
    print("\nLoading original data for duplicate checking...")
    original_train = load_csv(os.path.join(ORIGINAL_DIR, "train.csv"))
    original_test = load_csv(os.path.join(ORIGINAL_DIR, "test.csv"))
    original_val = load_csv(os.path.join(ORIGINAL_DIR, "validation.csv"))
    original_all = original_train + original_test + original_val
    
    print(f"  Original data: {len(original_all)} records")
    
    # Validation statistics
    stats = {
        'total': len(new_data),
        'valid_sequences': 0,
        'invalid_sequences': 0,
        'valid_gene_types': 0,
        'invalid_gene_types': 0,
        'duplicates': 0,
        'missing_fields': 0
    }
    
    validated_data = []
    invalid_data = []
    
    required_fields = ['NCBIGeneID', 'Symbol', 'Description', 'GeneType', 'NucleotideSequence']
    
    print("\nValidating records...")
    for i, row in enumerate(new_data):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(new_data)}...")
        
        # Check required fields
        missing = [f for f in required_fields if not row.get(f)]
        if missing:
            stats['missing_fields'] += 1
            invalid_data.append({**row, '_error': f'Missing fields: {missing}'})
            continue
        
        # Validate sequence
        seq = row.get('NucleotideSequence', '')
        is_valid_seq, cleaned_seq = validate_sequence(seq)
        
        if not is_valid_seq:
            stats['invalid_sequences'] += 1
            invalid_data.append({**row, '_error': 'Invalid sequence'})
            continue
        
        # Validate gene type
        gene_type = row.get('GeneType', '')
        if gene_type not in VALID_GENE_TYPES:
            stats['invalid_gene_types'] += 1
            invalid_data.append({**row, '_error': f'Invalid gene type: {gene_type}'})
            continue
        
        # Update row with cleaned sequence
        row['NucleotideSequence'] = cleaned_seq
        validated_data.append(row)
        stats['valid_sequences'] += 1
        stats['valid_gene_types'] += 1
    
    print(f"\nValidation Results:")
    print(f"  Valid records: {stats['valid_sequences']}")
    print(f"  Invalid records: {stats['invalid_sequences'] + stats['invalid_gene_types'] + stats['missing_fields']}")
    
    # Check duplicates
    print("\nChecking for duplicates...")
    duplicates, unique_data = check_duplicates(validated_data, original_all)
    stats['duplicates'] = len(duplicates)
    
    print(f"  Duplicates found: {stats['duplicates']}")
    print(f"  Unique new records: {len(unique_data)}")
    
    # Save validated data
    if unique_data:
        validated_file = os.path.join(OUTPUT_DIR, "additional_genes_validated.csv")
        fieldnames = required_fields + ['GeneGroupMethod']
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(validated_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(unique_data)
        
        print(f"\nSaved {len(unique_data)} validated records to: {validated_file}")
    
    # Generate report
    gene_type_counts = Counter(row.get('GeneType', '') for row in unique_data)
    
    report_lines = [
        "# New Data Validation Report",
        "",
        "## Summary",
        "",
        f"**Total Records**: {stats['total']}",
        f"**Valid Records**: {stats['valid_sequences']}",
        f"**Invalid Records**: {stats['invalid_sequences'] + stats['invalid_gene_types'] + stats['missing_fields']}",
        f"**Duplicates Removed**: {stats['duplicates']}",
        f"**Final Unique Records**: {len(unique_data)}",
        "",
        "## Validation Details",
        "",
        f"- Valid sequences: {stats['valid_sequences']}",
        f"- Invalid sequences: {stats['invalid_sequences']}",
        f"- Invalid gene types: {stats['invalid_gene_types']}",
        f"- Missing fields: {stats['missing_fields']}",
        "",
        "## Gene Type Distribution (Validated Data)",
        "",
        "| Gene Type | Count |",
        "|-----------|-------|"
    ]
    
    for gt, count in gene_type_counts.most_common():
        report_lines.append(f"| {gt} | {count} |")
    
    report_lines.append("")
    
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {REPORT_FILE}")


if __name__ == "__main__":
    validate_data()
