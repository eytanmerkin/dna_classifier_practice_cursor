# New Data Validation Report

## Status

No new data found to validate.

## Validation Criteria

When data is available, this script will validate:

1. **Sequence Format**:
   - ACGTN characters only
   - Length: 2-1000 nucleotides
   - No invalid characters

2. **Column Structure**:
   - Required columns: NCBIGeneID, Symbol, Description, GeneType, NucleotideSequence
   - Consistent with original data format

3. **Gene Types**:
   - Must be one of the known types
   - Valid values: PSEUDO, BIOLOGICAL_REGION, ncRNA, snoRNA, etc.

4. **Duplicates**:
   - Check against original dataset
   - Remove duplicates to prevent data leakage

5. **Data Quality**:
   - Complete records (no missing critical fields)
   - Valid gene IDs
