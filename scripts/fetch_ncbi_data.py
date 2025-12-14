"""
Fetch additional DNA sequences from NCBI.

This script queries NCBI Gene database for sequences matching specific
gene types, especially rare classes (scRNA, snRNA, rRNA).
"""

import os
import csv
import time
import sys
from collections import defaultdict

# Try to import biopython
try:
    from Bio import Entrez
    from Bio import SeqIO
    HAS_BIO = True
except ImportError:
    HAS_BIO = False
    print("Warning: biopython not installed. Install with: pip install biopython")
    print("This script will generate a template for manual data collection.")

OUTPUT_DIR = "DNA_seq_pred_additional"
REPORT_FILE = "analysis_output/ncbi_fetch_report.md"

# NCBI API rate limit: 3 requests per second
API_DELAY = 0.34  # seconds between requests


def setup_entrez(email="your.email@example.com"):
    """Set up Entrez API access."""
    if not HAS_BIO:
        return False
    Entrez.email = email
    return True


def search_genes(gene_type, organism="Homo sapiens", max_results=1000):
    """
    Search NCBI Gene database for genes of a specific type.
    
    Args:
        gene_type: Gene type to search for (e.g., 'ncRNA', 'snRNA')
        organism: Organism filter
        max_results: Maximum number of results to retrieve
        
    Returns:
        List of gene IDs
    """
    if not HAS_BIO:
        return []
    
    term = f'{gene_type}[Gene Type] AND {organism}[Organism]'
    
    print(f"Searching for: {term}")
    
    try:
        handle = Entrez.esearch(db="gene", term=term, retmax=max_results)
        record = Entrez.read(handle)
        gene_ids = record["IdList"]
        print(f"  Found {len(gene_ids)} genes")
        return gene_ids
    except Exception as e:
        print(f"  Error searching: {e}")
        return []


def get_gene_summary(gene_id):
    """Get summary information for a gene ID."""
    if not HAS_BIO:
        return None
    
    try:
        handle = Entrez.esummary(db="gene", id=gene_id)
        record = Entrez.read(handle)
        return record[0] if record else None
    except Exception as e:
        print(f"  Error fetching summary for {gene_id}: {e}")
        return None


def get_sequence_from_gene(gene_id, gene_summary):
    """
    Attempt to get sequence data from gene record.
    
    Note: This is a simplified approach. In practice, you may need to:
    1. Get RefSeq/GenBank accessions from gene record
    2. Fetch sequences from nucleotide database
    3. Extract relevant regions
    """
    # This is a placeholder - actual implementation would fetch sequences
    # from linked RefSeq/GenBank entries
    return None


def fetch_gene_data(gene_types, organism="Homo sapiens", max_per_type=500):
    """
    Fetch gene data for specified gene types.
    
    Args:
        gene_types: List of gene types to fetch
        organism: Organism filter
        max_per_type: Maximum genes per type
        
    Returns:
        List of gene records with metadata
    """
    if not HAS_BIO:
        print("\n" + "="*60)
        print("NCBI Data Fetching - Template Mode")
        print("="*60)
        print("\nbiopython is not installed. Generating template for manual collection.")
        print("\nTo install biopython:")
        print("  pip install biopython")
        print("\nThen set your email in this script and run again.")
        print("\n" + "="*60)
        return []
    
    all_genes = []
    
    print("\n" + "="*60)
    print("Fetching Data from NCBI")
    print("="*60)
    
    for gene_type in gene_types:
        print(f"\nProcessing: {gene_type}")
        
        # Search for genes
        gene_ids = search_genes(gene_type, organism, max_per_type)
        time.sleep(API_DELAY)
        
        # Fetch summaries (limited for demo)
        fetched = 0
        for gene_id in gene_ids[:min(100, len(gene_ids))]:  # Limit to 100 for demo
            summary = get_gene_summary(gene_id)
            time.sleep(API_DELAY)
            
            if summary:
                gene_record = {
                    'NCBIGeneID': gene_id,
                    'Symbol': summary.get('Name', ''),
                    'Description': summary.get('Description', ''),
                    'GeneType': gene_type,
                    'GeneGroupMethod': 'NCBI Query',
                    'NucleotideSequence': ''  # Would need additional API calls
                }
                all_genes.append(gene_record)
                fetched += 1
        
        print(f"  Fetched {fetched} gene records")
    
    return all_genes


def save_gene_data(genes, output_file):
    """Save gene data to CSV file."""
    if not genes:
        print("No data to save.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fieldnames = ['NCBIGeneID', 'Symbol', 'Description', 'GeneType', 
                  'GeneGroupMethod', 'NucleotideSequence']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(genes)
    
    print(f"\nSaved {len(genes)} records to {output_file}")


def generate_fetch_report(genes_fetched, gene_types):
    """Generate a report on the data fetching process."""
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    
    type_counts = defaultdict(int)
    for gene in genes_fetched:
        type_counts[gene['GeneType']] += 1
    
    report_lines = [
        "# NCBI Data Fetching Report",
        "",
        "## Summary",
        "",
        f"**Total Genes Fetched**: {len(genes_fetched)}",
        "",
        "## Gene Type Distribution",
        "",
        "| Gene Type | Count |",
        "|-----------|-------|"
    ]
    
    for gt in gene_types:
        count = type_counts[gt]
        report_lines.append(f"| {gt} | {count} |")
    
    report_lines.extend([
        "",
        "## Notes",
        "",
        "### Current Limitations:",
        "",
        "1. **Sequence Data**: This script fetches gene metadata but not sequences.",
        "   To get sequences, you need to:",
        "   - Extract RefSeq/GenBank accessions from gene records",
        "   - Query nucleotide database for sequences",
        "   - Extract relevant sequence regions",
        "",
        "2. **Rate Limiting**: NCBI limits to 3 requests/second.",
        "   Large queries will take significant time.",
        "",
        "3. **Data Completeness**: Not all genes have complete annotations.",
        "",
        "### Next Steps:",
        "",
        "1. Install biopython: `pip install biopython`",
        "2. Set email in script: `Entrez.email = 'your.email@example.com'`",
        "3. Extend script to fetch actual sequences",
        "4. Or use NCBI Datasets CLI for bulk downloads",
        ""
    ])
    
    with open(REPORT_FILE, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {REPORT_FILE}")


def main():
    """Main function to fetch NCBI data."""
    
    # Focus on rare gene types
    target_gene_types = ['scRNA', 'snRNA', 'rRNA', 'tRNA', 'snoRNA', 'ncRNA']
    
    # Set up API (user should set their email)
    if HAS_BIO:
        email = os.getenv('NCBI_EMAIL', 'your.email@example.com')
        if email == 'your.email@example.com':
            print("Warning: Please set NCBI_EMAIL environment variable or modify script")
            print("NCBI requires an email for API access.")
        setup_entrez(email)
    
    # Fetch data
    genes = fetch_gene_data(target_gene_types, max_per_type=200)
    
    # Save data
    if genes:
        output_file = os.path.join(OUTPUT_DIR, "additional_genes.csv")
        save_gene_data(genes, output_file)
    
    # Generate report
    generate_fetch_report(genes, target_gene_types)
    
    if not HAS_BIO:
        print("\n" + "="*60)
        print("INSTALLATION REQUIRED")
        print("="*60)
        print("\nTo use this script, install biopython:")
        print("  pip install biopython")
        print("\nThen set your email:")
        print("  export NCBI_EMAIL='your.email@example.com'")
        print("  # or modify the script directly")
        print("\n" + "="*60)


if __name__ == "__main__":
    main()
