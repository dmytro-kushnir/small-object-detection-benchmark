# Camponotus Raw Data Layout

Raw assets for Camponotus dataset creation.

## Structure

- `in_situ/`
  - `seq_001/`, `seq_002/`, ...
  - frame sequences extracted from your ant-farm recordings
- `external/images/`
  - internet images used as additional training data
- `metadata/`
  - source notes, license references, capture conditions, and mapping files

## Rules

1. Preserve frame order inside every `in_situ/seq_*` directory.
2. Keep external images separate from in-situ data.
3. Do not place annotations directly in this raw directory.
4. Keep provenance in `metadata/` for reproducibility.

Processing outputs should be written to `datasets/camponotus_processed/`.
