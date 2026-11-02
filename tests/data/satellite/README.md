# Satellite compatibility fixtures

The footprint file `footprints/GOSAT-BRAZIL-column_SOUTHAMERICA_201601.nc`
contains three January 1 soundings, rather than the original full month.
The column, flux and boundary-condition files are unchanged. The column file
contains all three selected times; the BC file still represents January's
monthly coverage. The monthly footprint filename is retained for the ACRG
source-format reader.

## Footprint provenance

Source: the same file at repository commit
`7973c2dc4f5e358b9d3eb14a0b90c24ce7761214`, SHA256
`da17401f96dd33bcd884d43a4669eea2dbf9608b93bb18ab740341993299d394`.
It can be recovered with `git show <commit>:tests/data/satellite/footprints/GOSAT-BRAZIL-column_SOUTHAMERICA_201601.nc`.

Generated on 2026-10-21 with netCDF4 1.7.2 by selecting zero-based time indices
`[0, 18, 35]`, the first, middle and last of the 36 January 1 times shared with
the bundled GOSAT column file:

- `2016-01-01T14:59:12.500`
- `2016-01-01T16:40:42`
- `2016-01-01T16:43:46`

The copy used `netCDF4.Dataset`, with automatic masking/scaling and character
conversion disabled. Each time-dependent variable was indexed on its existing
time axis; other variables were copied in full. Dimension and variable order,
dtypes, all global/variable attributes, compression settings and the complete
357-by-190 spatial grid and 20 particle-height levels were retained. Chunk
lengths were capped to the reduced dimensions. Attributes were assigned after
the numeric data to avoid quantizing already quantized source values again.

Before replacing the source file, generation checked every retained value in
all 17 variables against that source selection with exact array equality, plus
all dimensions, dtypes and global/variable attributes. Only the time axis and
storage chunk sizes changed. This fixture covers input-format compatibility;
full-month ingestion volume is not an Inversions scientific requirement.
