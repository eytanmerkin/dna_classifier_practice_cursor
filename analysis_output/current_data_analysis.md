# Current Data Structure Analysis

**Total Samples**: 35496
- Train: 22593
- Validation: 4577
- Test: 8326

## NCBI Gene ID Analysis

**Total Gene IDs**: 35496
**Unique Gene IDs**: 23779
**Duplicate IDs**: 11717

### ID Patterns:
- Numeric IDs: 35496
- Non-numeric IDs: 0

**Note**: NCBI Gene IDs are typically numeric. Non-numeric IDs may indicate
LOC identifiers or other annotation types.

### Sample Gene IDs:

- 121627833
- 122149496
- 100616310
- 121627970
- 100616437
- 106479354
- 122149491
- 106481979
- 100189358
- 106479848
- 406929
- 105379216
- 406955
- 100189518
- 122094898
- 406961
- 108281127
- 124310610
- 124906597
- 100874348

## Gene Type Distribution

| GeneType | Count | Percentage |
|----------|-------|------------|
| PSEUDO | 16153 | 45.51% |
| BIOLOGICAL_REGION | 10974 | 30.92% |
| ncRNA | 3907 | 11.01% |
| snoRNA | 1792 | 5.05% |
| PROTEIN_CODING | 809 | 2.28% |
| tRNA | 708 | 1.99% |
| OTHER | 587 | 1.65% |
| rRNA | 357 | 1.01% |
| snRNA | 205 | 0.58% |
| scRNA | 4 | 0.01% |

## Sequence Length Statistics

- **Min**: 2
- **Max**: 1000
- **Mean**: 361.47
- **Median**: 295

## Symbol Analysis

**Total Symbols**: 35496
**Unique Symbols**: 23690

## GeneGroupMethod

**Unique Methods**: 1
**Values**: NCBI Ortholog

## Data Quality Observations

### For NCBI Query Strategy:

1. **Gene IDs**: Most are numeric NCBI Gene IDs, suitable for direct queries
2. **Gene Types**: Well-defined categories matching NCBI annotations
3. **Sequence Length**: Range 2-1000 nucleotides (good for filtering)
4. **Metadata**: Rich metadata available (Symbol, Description, GeneType)

## Recommendations for Data Collection

### Query Strategy:

1. **Use Gene Type Filters**:
   - Query NCBI Gene database with `[Gene Type]` filter
   - Focus on rare types: scRNA, snRNA, rRNA

2. **Sequence Length Filter**:
   - Filter results to 2-1000 nucleotide range
   - Match current data distribution

3. **Organism Filter** (if applicable):
   - Determine organism from current data
   - Apply organism filter to maintain consistency

4. **Quality Filters**:
   - Prefer RefSeq over GenBank
   - Require complete annotations

### Expected Data Format:

New data should match existing columns:
- `NCBIGeneID`: Numeric or LOC identifier
- `Symbol`: Gene symbol
- `Description`: Gene description
- `GeneType`: One of the known types
- `GeneGroupMethod`: 'NCBI Ortholog' (or similar)
- `NucleotideSequence`: DNA sequence (ACGTN only)
