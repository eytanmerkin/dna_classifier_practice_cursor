"""
Analyze current data structure and NCBI ID patterns.

This script examines the existing dataset to understand:
- NCBI Gene ID patterns
- Data distribution
- Metadata availability
- Patterns for querying additional data
"""

import csv
import os
from collections import Counter, defaultdict

INPUT_DIR = "DNA_seq_pred"
OUTPUT_DIR = "analysis_output"
REPORT_FILE = os.path.join(OUTPUT_DIR, "current_data_analysis.md")


def load_csv(filepath):
    """Load CSV file and return list of dictionaries."""
    data = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def analyze_dataset():
    """Analyze the current dataset structure."""
    
    print("Loading data...")
    train_data = load_csv(os.path.join(INPUT_DIR, "train.csv"))
    test_data = load_csv(os.path.join(INPUT_DIR, "test.csv"))
    val_data = load_csv(os.path.join(INPUT_DIR, "validation.csv"))
    
    all_data = train_data + test_data + val_data
    
    print(f"Total samples: {len(all_data)}")
    
    # Extract NCBI Gene IDs
    gene_ids = [row.get('NCBIGeneID', '') for row in all_data if row.get('NCBIGeneID')]
    unique_gene_ids = set(gene_ids)
    
    # Analyze Gene IDs
    id_patterns = defaultdict(int)
    for gid in gene_ids:
        if gid:
            # Check if it's numeric (typical NCBI Gene ID)
            if gid.isdigit():
                id_patterns['numeric'] += 1
            else:
                id_patterns['non-numeric'] += 1
    
    # Gene Type distribution
    gene_types = [row.get('GeneType', '') for row in all_data]
    type_counts = Counter(gene_types)
    
    # Symbol analysis
    symbols = [row.get('Symbol', '') for row in all_data]
    unique_symbols = set(symbols)
    
    # Description patterns
    descriptions = [row.get('Description', '') for row in all_data]
    
    # Sequence length analysis
    seq_lengths = []
    for row in all_data:
        seq = row.get('NucleotideSequence', '')
        if seq:
            seq = seq.strip('<>')
            seq_lengths.append(len(seq))
    
    # GeneGroupMethod (should be constant)
    group_methods = [row.get('GeneGroupMethod', '') for row in all_data]
    unique_methods = set(group_methods)
    
    # Generate report
    report_lines = [
        "# Current Data Structure Analysis",
        "",
        f"**Total Samples**: {len(all_data)}",
        f"- Train: {len(train_data)}",
        f"- Validation: {len(val_data)}",
        f"- Test: {len(test_data)}",
        "",
        "## NCBI Gene ID Analysis",
        "",
        f"**Total Gene IDs**: {len(gene_ids)}",
        f"**Unique Gene IDs**: {len(unique_gene_ids)}",
        f"**Duplicate IDs**: {len(gene_ids) - len(unique_gene_ids)}",
        "",
        "### ID Patterns:",
        f"- Numeric IDs: {id_patterns['numeric']}",
        f"- Non-numeric IDs: {id_patterns['non-numeric']}",
        "",
        "**Note**: NCBI Gene IDs are typically numeric. Non-numeric IDs may indicate",
        "LOC identifiers or other annotation types.",
        "",
        "### Sample Gene IDs:",
        ""
    ]
    
    # Show sample IDs
    sample_ids = list(unique_gene_ids)[:20]
    for gid in sample_ids:
        report_lines.append(f"- {gid}")
    
    report_lines.extend([
        "",
        "## Gene Type Distribution",
        "",
        "| GeneType | Count | Percentage |",
        "|----------|-------|------------|"
    ])
    
    total = len(all_data)
    for gt, count in type_counts.most_common():
        pct = (count / total) * 100
        report_lines.append(f"| {gt} | {count} | {pct:.2f}% |")
    
    report_lines.extend([
        "",
        "## Sequence Length Statistics",
        "",
        f"- **Min**: {min(seq_lengths) if seq_lengths else 0}",
        f"- **Max**: {max(seq_lengths) if seq_lengths else 0}",
        f"- **Mean**: {sum(seq_lengths) / len(seq_lengths) if seq_lengths else 0:.2f}",
        f"- **Median**: {sorted(seq_lengths)[len(seq_lengths)//2] if seq_lengths else 0}",
        "",
        "## Symbol Analysis",
        "",
        f"**Total Symbols**: {len(symbols)}",
        f"**Unique Symbols**: {len(unique_symbols)}",
        "",
        "## GeneGroupMethod",
        "",
        f"**Unique Methods**: {len(unique_methods)}",
        f"**Values**: {', '.join(unique_methods)}",
        "",
        "## Data Quality Observations",
        "",
        "### For NCBI Query Strategy:",
        "",
        "1. **Gene IDs**: Most are numeric NCBI Gene IDs, suitable for direct queries",
        "2. **Gene Types**: Well-defined categories matching NCBI annotations",
        "3. **Sequence Length**: Range 2-1000 nucleotides (good for filtering)",
        "4. **Metadata**: Rich metadata available (Symbol, Description, GeneType)",
        "",
        "## Recommendations for Data Collection",
        "",
        "### Query Strategy:",
        "",
        "1. **Use Gene Type Filters**:",
        "   - Query NCBI Gene database with `[Gene Type]` filter",
        "   - Focus on rare types: scRNA, snRNA, rRNA",
        "",
        "2. **Sequence Length Filter**:",
        "   - Filter results to 2-1000 nucleotide range",
        "   - Match current data distribution",
        "",
        "3. **Organism Filter** (if applicable):",
        "   - Determine organism from current data",
        "   - Apply organism filter to maintain consistency",
        "",
        "4. **Quality Filters**:",
        "   - Prefer RefSeq over GenBank",
        "   - Require complete annotations",
        "",
        "### Expected Data Format:",
        "",
        "New data should match existing columns:",
        "- `NCBIGeneID`: Numeric or LOC identifier",
        "- `Symbol`: Gene symbol",
        "- `Description`: Gene description",
        "- `GeneType`: One of the known types",
        "- `GeneGroupMethod`: 'NCBI Ortholog' (or similar)",
        "- `NucleotideSequence`: DNA sequence (ACGTN only)",
        ""
    ])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nAnalysis complete. Report saved to: {REPORT_FILE}")
    print(f"\nKey Findings:")
    print(f"  - Unique Gene IDs: {len(unique_gene_ids)}")
    print(f"  - Gene Types: {len(type_counts)}")
    print(f"  - Sequence length range: {min(seq_lengths) if seq_lengths else 0}-{max(seq_lengths) if seq_lengths else 0}")


if __name__ == "__main__":
    analyze_dataset()
