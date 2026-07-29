# G4 development serialization recovery

## Preserved technical failure

Development job `18213953` ran seed 731 for execution source
`189427d5ccca9187618ab8be1cc2cf7d7105b216` on `bp1-compute095`.
It used the substantive G3 controls \(C=1024\), \(P=64\), completed in
55 minutes 51 seconds, used 1,996 MiB Slurm MaxRSS, and exited `1:0`.

All expensive calculations completed before publication.  The create-only
partial stage contains:

- `projected_locations.npy`, SHA-256
  `aec20f2d3fd1c93c6ba52c2fbc4a84986121debafbad48e3b7e07943911a33a7`;
- `log_likelihood.npy`, SHA-256
  `500772c940b5804db42f878159be3996392aae564ff00758da03db18aee9b74f`;
- `resource.time`, SHA-256
  `1306c45706f40a1ce94374f491c7a4c22464052e90fe0635717eaa821a391ff0`;
  and
- the preserved stderr traceback, SHA-256
  `5afd852aa62241797d5ceaaae28972ce5d285b2b64a628290249eecfcd410541`.

The projected-bank file is byte-identical to the selected G3 reference.
There is no `seed_report.json` or `G4_SEED_COMPLETE.txt`.

The failure occurred only in final JSON serialization.  NumPy 2.2 returns a
`numpy.bool` scalar for the translation-tolerance comparison; strict standard
library JSON serialization rejected it.  This is not an approximation,
resource, or scientific failure.

## Recovery

The repair converts the translation tolerance to a Python `float` and its
comparison to a Python `bool` at the point where the diagnostic is formed.
It does not broaden the canonical JSON encoder or change previously
serializable artifact identities.  A focused regression requires strict
canonical serialization of the translation-parity record.

The failed directory remains unchanged.  One retry uses a distinct
`seed731_retry1` directory.  The launcher accepts that explicit create-only
stage, records the repair Git SHA in the seed report, and authenticates the
unchanged threshold supplement from the original execution source.  The
scientific execution revision, input, grid, spectrum, G3 controls, seed,
sample/rank ladders, and thresholds remain unchanged.
