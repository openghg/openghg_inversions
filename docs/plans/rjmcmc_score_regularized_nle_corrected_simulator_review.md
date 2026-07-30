# Corrected score NLE simulator and execution review

## Scope

Independent read-only review of the corrected PCG64 simulator, replay and
domain-separation tests, private initializer/optimizer stream construction,
and committed Slurm wrappers. The review did not modify scientific code.

## Initial findings and disposition

The simulator correctly constructs allocation, root total, and Gaussian noise
from distinct case/domain/role-keyed PCG64 streams. Its v2 evidence binds the
latent uniforms, transforms, construction method, array hashes, allocation
artifact, and NumPy/SciPy versions.

The initial E2 implementation was held because:

- replay/prefix tests did not cover every public domain;
- domain separation did not cover every frozen case;
- private stream uniqueness was reported but not enforced before fitting;
- wrappers did not load the pinned Git module and discarded scheduler output
  before their preserved task logs existed.

All findings were corrected. Tests now cover every six-case/three-domain
combination and the complete private role catalogue. Each attempt fails before
model construction unless private PCG64 streams are mutually unique and
disjoint from every simulator stream, and unless derived private JAX seeds are
unique. Wrappers load `git/2.45.1-pqk5`, preserve default scheduler logs, use
shared nodes, and make the Python driver the terminal `exec` action.

## Final verdict

**PASS for commit, push, and launch.**

Launch must use a fresh clean detached full-SHA worktree and a fresh run root.
The committed wrappers enforce this condition. The reviewed request of one
CPU, 8 GiB, and one hour on a shared node is reasonable for the initial
oracle/compile canary and must be revised from measured accounting evidence
before larger work if appropriate.
