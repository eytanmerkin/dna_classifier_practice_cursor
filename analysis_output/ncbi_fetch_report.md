# NCBI Data Fetching Report

## Summary

**Total Genes Fetched**: 0

## Gene Type Distribution

| Gene Type | Count |
|-----------|-------|
| scRNA | 0 |
| snRNA | 0 |
| rRNA | 0 |
| tRNA | 0 |
| snoRNA | 0 |
| ncRNA | 0 |

## Notes

### Current Limitations:

1. **Sequence Data**: This script fetches gene metadata but not sequences.
   To get sequences, you need to:
   - Extract RefSeq/GenBank accessions from gene records
   - Query nucleotide database for sequences
   - Extract relevant sequence regions

2. **Rate Limiting**: NCBI limits to 3 requests/second.
   Large queries will take significant time.

3. **Data Completeness**: Not all genes have complete annotations.

### Next Steps:

1. Install biopython: `pip install biopython`
2. Set email in script: `Entrez.email = 'your.email@example.com'`
3. Extend script to fetch actual sequences
4. Or use NCBI Datasets CLI for bulk downloads
