import csv
import os
import math
import statistics
from collections import Counter, defaultdict

OUTPUT_DIR = "analysis_output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORT_FILE = os.path.join(OUTPUT_DIR, "report.md")
HTML_FILE = os.path.join(OUTPUT_DIR, "visualizations.html")

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def load_csv(filepath):
    data = []
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def get_column(data, col_name):
    return [row.get(col_name) for row in data]

def calculate_correlations(data, col1, col2):
    # Cramer's V approximation for categorical data is complex without pandas/scipy
    # We will use simple cross-tabulation observation
    counts = defaultdict(int)
    for row in data:
        val1 = row.get(col1, 'N/A')
        val2 = row.get(col2, 'N/A')
        counts[(val1, val2)] += 1
    return counts

def analyze_dataset():
    print("Loading data using standard library...")
    try:
        train_data = load_csv("DNA_seq_pred/train.csv")
        test_data = load_csv("DNA_seq_pred/test.csv")
        val_data = load_csv("DNA_seq_pred/validation.csv")
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return

    # Add split info
    for row in train_data: row['split'] = 'train'
    for row in test_data: row['split'] = 'test'
    for row in val_data: row['split'] = 'val'
    
    all_data = train_data + test_data + val_data
    
    # 1. Summary
    report_lines = []
    report_lines.append("# Data Analysis Report\n")
    report_lines.append("## 1. Summary of Variables and Labels (3.3.1)\n")
    report_lines.append(f"**Total Samples**: {len(all_data)}")
    report_lines.append(f"- Train: {len(train_data)}")
    report_lines.append(f"- Validation: {len(val_data)}")
    report_lines.append(f"- Test: {len(test_data)}\n")

    # Columns
    if all_data:
        columns = list(all_data[0].keys())
        report_lines.append("### Columns:\n")
        for col in columns:
            if col == 'split': continue
            values = get_column(all_data, col)
            unique_vals = set(values)
            report_lines.append(f"- **{col}**: {len(unique_vals)} unique values")

    # Target Distribution
    gene_types = get_column(all_data, 'GeneType')
    type_counts = Counter(gene_types)
    report_lines.append("\n### Target Variable Analysis (GeneType):")
    report_lines.append("| GeneType | Count |")
    report_lines.append("| --- | --- |")
    for gt, count in type_counts.most_common():
        report_lines.append(f"| {gt} | {count} |")

    # 2. Correlations/Redundant
    report_lines.append("\n## 2. Correlations and Redundant Variables (3.3.2)\n")
    
    constant_vars = []
    high_card_vars = []
    
    for col in columns:
        if col == 'split': continue
        values = get_column(all_data, col)
        unique_vals = set(values)
        if len(unique_vals) <= 1:
            constant_vars.append(col)
        if len(unique_vals) > len(all_data) * 0.9:
            high_card_vars.append(col)
            
    if constant_vars:
        report_lines.append(f"**Redundant (Constant) Variables**: {', '.join(constant_vars)}")
    else:
        report_lines.append("**Redundant (Constant) Variables**: None found.")
        
    report_lines.append(f"**High Cardinality Variables (Likely IDs)**: {', '.join(high_card_vars)}\n")

    # Sequence Analysis
    # Clean sequences (remove < and >)
    for row in all_data:
        row['NucleotideSequence'] = row['NucleotideSequence'].strip('<>')

    seq_lengths = [len(row['NucleotideSequence']) for row in all_data]
    avg_len = statistics.mean(seq_lengths)
    max_len = max(seq_lengths)
    min_len = min(seq_lengths)
    
    report_lines.append(f"**Sequence Lengths**: Min={min_len}, Max={max_len}, Mean={avg_len:.2f}")

    # 3. Problems
    report_lines.append("\n## 3. Data Problems and Fixes (3.3.3)\n")
    
    # Missing values
    missing_counts = defaultdict(int)
    for row in all_data:
        for col in columns:
            if not row.get(col):
                missing_counts[col] += 1
                
    if missing_counts:
        report_lines.append("### Missing Values:")
        for col, count in missing_counts.items():
            report_lines.append(f"- {col}: {count} missing")
    else:
        report_lines.append("### Missing Values: None detected.")

    # Duplicates
    sequences = get_column(all_data, 'NucleotideSequence')
    seq_counts = Counter(sequences)
    duplicates = [seq for seq, count in seq_counts.items() if count > 1]
    total_dupes = sum(seq_counts[seq] for seq in duplicates) - len(duplicates)
    
    report_lines.append(f"\n### Duplicate Sequences: {len(duplicates)} unique sequences appear multiple times (Total {total_dupes} duplicate entries).")
    
    # Leakage
    train_seqs = set(get_column(train_data, 'NucleotideSequence'))
    test_seqs = set(get_column(test_data, 'NucleotideSequence'))
    leakage = train_seqs.intersection(test_seqs)
    
    if leakage:
        report_lines.append(f"- **CRITICAL**: {len(leakage)} sequences found in both Train and Test sets (Data Leakage).")
        report_lines.append("  - **Fix**: Remove these sequences from the test set.")
    
    # Invalid chars
    invalid_count = 0
    valid_chars = set('ACGTN')
    for seq in sequences:
        if not set(seq.upper()).issubset(valid_chars):
            invalid_count += 1
            
    report_lines.append(f"\n### Invalid Characters: {invalid_count} sequences with non-ACGTN characters.")

    # 4. Summary
    report_lines.append("\n## 4. Summary of Conclusions (3.3.4)\n")
    report_lines.append(f"1. Dataset has {len(all_data)} samples.")
    report_lines.append(f"2. {len(type_counts)} classes in GeneType. Top class: {type_counts.most_common(1)[0]}")
    report_lines.append(f"3. Sequence length range: {min_len}-{max_len}.")
    if leakage:
        report_lines.append(f"4. **Action Required**: Fix data leakage of {len(leakage)} samples.")

    # Save Report
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(report_lines))

    # Generate HTML for Visualizations
    generate_html(type_counts, seq_lengths, gene_types, seq_lengths)

    print(f"Analysis complete. Report saved to {REPORT_FILE}")

def generate_html(type_counts, seq_lengths, gene_types, all_seq_lengths):
    labels = list(type_counts.keys())
    data = list(type_counts.values())
    
    # Histogram data preparation
    hist_bins = list(range(0, max(seq_lengths)+100, 100))
    hist_data = [0] * (len(hist_bins)-1)
    for l in seq_lengths:
        for i in range(len(hist_bins)-1):
            if hist_bins[i] <= l < hist_bins[i+1]:
                hist_data[i] += 1
                break
    
    html_content = f"""
    <html>
    <head>
        <title>Data Analysis Visualizations</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: sans-serif; padding: 20px; }}
            .chart-container {{ width: 800px; margin: 20px auto; }}
        </style>
    </head>
    <body>
        <h1>Data Visualizations</h1>
        
        <div class="chart-container">
            <h2>GeneType Distribution</h2>
            <canvas id="typeChart"></canvas>
        </div>
        
        <div class="chart-container">
            <h2>Sequence Length Distribution</h2>
            <canvas id="lenChart"></canvas>
        </div>

        <script>
            const ctx1 = document.getElementById('typeChart').getContext('2d');
            new Chart(ctx1, {{
                type: 'bar',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        label: 'Count',
                        data: {data},
                        backgroundColor: 'rgba(54, 162, 235, 0.5)'
                    }}]
                }}
            }});

            const ctx2 = document.getElementById('lenChart').getContext('2d');
            new Chart(ctx2, {{
                type: 'bar',
                data: {{
                    labels: {hist_bins[:-1]},
                    datasets: [{{
                        label: 'Sequence Lengths',
                        data: {hist_data},
                        backgroundColor: 'rgba(255, 99, 132, 0.5)'
                    }}]
                }}
            }});
        </script>
    </body>
    </html>
    """
    
    with open(HTML_FILE, "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    analyze_dataset()
