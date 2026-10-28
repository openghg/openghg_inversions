# Bounded centered-boundary memory experiment

These research scripts leave the production precision guard and APIs unchanged.
They compare four otherwise identical collapsed centered-boundary recipes:

* `float64`: active boundary design and numerical work remain float64.
* `active_b32`: round active boundary design to float32 once, store it as float32
  in the prediction graph, and widen those rounded values for the numerical cache.
* `blocked64`: float64 baseline with the residual and 32 design-column blocks
  transformed into a preallocated float64 right-hand side.
* `combined`: both changes.

Boundary deviations, fixed intercepts, reference shifts, all covariance factors,
cache matrices and solve arithmetic remain float64. The blocked constructor
retains the original validation and input copies; refresh is unchanged.
`precision_probe.py` preserves the preparation and recipe hooks of the previous
PR986 controlled precision experiment (source script
SHA256 `c9553085e14e0aa8d8ac04614e5349f6a21d252c0e0ffeafebe1617b90bb0bdb`),
with multi-site selection and independent OU noise by species/site. It retains
projection shapes rather than correctness-only float64 matrix copies.

Run from the repository root with its interpreter and `PYTHONPATH=.`. These are
private-data research scripts: the archive paths are explicit in preparation.
The MHD workload contains 129 rows/channel; MHD/BIR/BSD/TAC contains 618
rows/channel, retaining the same 304-state flux prior and selected full joint
aggregation covariance, including cross-site/channel terms. Rows are neither
replicated nor extrapolated. Corrected boundary operands are regenerated from
the pinned raw footprints/curtains and checked against the OPE231 oxygen oracle.

```sh
python scripts/research/centered_boundary_memory/selfcheck.py
python scripts/research/centered_boundary_memory/prepare_inputs.py --sites MHD --out build/memory-inputs/mhd
python scripts/research/centered_boundary_memory/prepare_inputs.py --sites MHD BIR BSD TAC --out build/memory-inputs/four_site
python scripts/research/centered_boundary_memory/measure.py check --config build/memory-inputs/mhd/config.json --variant float64 --out build/memory-checks/mhd/float64
python scripts/research/centered_boundary_memory/measure.py measure --config build/memory-inputs/mhd/config.json --variant float64 --chains 4 --out build/memory-results/mhd/float64/4
```

Preparation and all four correctness variants for both workloads must pass in a
separate job before the 16 measurement cases (2 workloads × 4 variants × 1/4
chains). Each measurement case runs in a fresh Python process. Four-chain cases
use actual PyMC multiprocessing with `cores=4` and `mp_ctx=spawn`. Set all BLAS
and numerical thread counts to one. Initial resources: four CPUs, 8 GiB, two
hours per case; register each potentially long Slurm submission with slurm-wakeup.

Correctness compilation and direct likelihood oracles never run after measured
sampling. The bounded 250 tune + 250 retained draws per chain support timing and
memory comparisons; they do not establish posterior or scientific acceptance.
Save timing includes reopening and checking the posterior artifact.

`result.json` records phase times and 100 ms sampled whole-process-tree PSS,
including spawned workers. Raw samples stream to `memory_samples.jsonl` to keep
telemetry memory bounded. PSS apportions shared pages; RSS is never summed.
Kernel cgroup current and cumulative peak are recorded separately because the
kernel peak cannot be assigned to individual phases. Sampling may miss brief
allocation spikes; preparation/import allocations can dominate whole-job peaks.
The phase array inventory reports actual dtypes/shapes and unique ndarray base
storage, prepared inputs, covariance modes/factors, caches, graph shared values,
compiled graph constants/storage, and sampler state. Traversal stops at 12
object levels; external-buffer aliases and allocator/native storage are subject
to the limits in the result. Whole-process PSS is the memory evidence.

Input manifests freeze source/spec/oracle/payload hashes; result manifests freeze
config/script hashes, dependency versions, Git revision and thread/runtime
settings. The noise and prior policy is explicit technical benchmark policy,
not an approved campaign model. Results do not by themselves remove the float64
production restriction.
