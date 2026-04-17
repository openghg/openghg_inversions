# PyMC and inversion pipeline refactor plan

Draft wiki content for [issue #392](https://github.com/openghg/openghg_inversions/issues/392).

This page is intended to do three things:

1. summarise what has already changed in `openghg_inversions`,
2. show how the main refactor threads fit together,
3. give a short map from legacy/temporary paths to their intended replacements.

## Executive summary

The current work is best understood as two umbrella refactors plus two related future capability tracks:

- [#370](https://github.com/openghg/openghg_inversions/issues/370): refactor PyMC model construction so the current fixed-basis inversion model is built from reusable functions and canonical xarray inputs;
- [#360](https://github.com/openghg/openghg_inversions/issues/360): refactor basis-function handling and output plumbing so basis objects can be retained, saved, and eventually used directly in post-processing;
- [#359](https://github.com/openghg/openghg_inversions/pull/359) and [#337](https://github.com/openghg/openghg_inversions/issues/337): high-resolution inner-domain / 6 km support, which currently collides with the refactors above and should be aligned with them rather than merged independently;
- [#338](https://github.com/openghg/openghg_inversions/issues/338): incorporation of CO2 / multi-sector work, likely drawing on <https://github.com/brendan-m-murphy/rhime-co2-inversions>.

The main PyMC refactor is already well advanced:

- Stage A ([#378](https://github.com/openghg/openghg_inversions/pull/378)) added regression coverage and extracted model construction from `inferpymc`.
- Stage B ([#380](https://github.com/openghg/openghg_inversions/pull/380)) let `inferpymc` consume xarray-based inversion inputs.
- Stage C ([#382](https://github.com/openghg/openghg_inversions/pull/382)) introduced `openghg_inversions.models` and reusable model components.
- Stage D ([#389](https://github.com/openghg/openghg_inversions/pull/389)) made the new builder the active runtime path.
- Stage E follow-up tidy-up ([#391](https://github.com/openghg/openghg_inversions/pull/391)) is making the modern/legacy boundary explicit rather than mixing the two concerns.

The main basis/output refactor is partly complete:

- `BasisFunctions` / `BasisOperator` were introduced in [#358](https://github.com/openghg/openghg_inversions/pull/358).
- Wrapper/runtime retention and DataTree save support were added in [#367](https://github.com/openghg/openghg_inversions/pull/367).
- A compatibility post-processing output mode was added in [#361](https://github.com/openghg/openghg_inversions/pull/361).
- `fixedbasisMCMC` output handling was cleaned up in [#390](https://github.com/openghg/openghg_inversions/pull/390).

The practical short-term direction is:

- keep shrinking the legacy `inferpymc` compatibility surface,
- add a parallel modern inversion pathway that works directly with canonical inputs and `InferenceData`,
- avoid landing major new model variants only on the legacy path,
- prioritise incorporation of multi-sector code from the CO2 work once the modern path is ready.

## How the workstreams fit together

### 1. PyMC model refactor (`#370`)

The goal of [#370](https://github.com/openghg/openghg_inversions/issues/370) is not to add a new scientific model directly. It is to separate model construction, sampling, and compatibility logic so the current fixed-basis model can be rebuilt cleanly and future models can be added with less duplication.

Completed and in-progress stages:

| Stage | Issue / PR | Status | Main result |
| --- | --- | --- | --- |
| A | [#371](https://github.com/openghg/openghg_inversions/issues/371) / [#378](https://github.com/openghg/openghg_inversions/pull/378) | done | Added regression tests and extracted `build_inferpymc_model(...)` from `inferpymc(...)`. |
| B | [#372](https://github.com/openghg/openghg_inversions/issues/372) / [#380](https://github.com/openghg/openghg_inversions/pull/380) | done | Added dataset/xarray-native inputs for `inferpymc(...)`. |
| C | [#373](https://github.com/openghg/openghg_inversions/issues/373) / [#382](https://github.com/openghg/openghg_inversions/pull/382) | done | Introduced `openghg_inversions.models` with reusable coords, priors, and component helpers. |
| D | [#374](https://github.com/openghg/openghg_inversions/issues/374) / [#389](https://github.com/openghg/openghg_inversions/pull/389) | done | Switched the active runtime to the new builder and dataset-first path. |
| E | [#375](https://github.com/openghg/openghg_inversions/issues/375) / [#391](https://github.com/openghg/openghg_inversions/pull/391) | open PR | Tidies the modern/legacy boundary and keeps `InferenceData` central. |

Key architectural direction after Stage D/E:

- canonical inversion inputs come from `make_inv_inputs(...)`,
- model construction should use canonical dims/coords (`region`, `bc_region`, etc.),
- modern sampling should work in terms of `InferenceData`,
- legacy naming and dict-shaped results should be confined to explicit compatibility adapters.

### 2. Basis functions and output refactor (`#360`)

[Issue #360](https://github.com/openghg/openghg_inversions/issues/360) is the parallel basis/output workstream. It matters because future multi-sector and inner-domain workflows need richer basis objects than the legacy flat-basis path.

Completed and planned slices:

| Slice | PR / issue | Status | Main result |
| --- | --- | --- | --- |
| Basis objects introduced | [#358](https://github.com/openghg/openghg_inversions/pull/358) | done | Added `BasisFunctions` and `BasisOperator`, including DataTree support and multi-source-aware operators. |
| Wrapper retention / dual-format save | [#367](https://github.com/openghg/openghg_inversions/pull/367) | done | `basis_functions_wrapper` can now optionally return basis objects and save DataTree artifacts. |
| Legacy-format post-processing reproduction | [#361](https://github.com/openghg/openghg_inversions/pull/361) | done | New post-processing code can reproduce the legacy HBMCMC-style output format. |
| `fixedbasisMCMC` output cleanup | [#376](https://github.com/openghg/openghg_inversions/issues/376) / [#390](https://github.com/openghg/openghg_inversions/pull/390) | done | Split output handling into clearer stages without changing inversion math. |
| Operator-backed post-processing default | `#360` PR-4/5 | not done | Still to come; current post-processing still retains legacy reconstruction in places. |

### 3. High-resolution inner-region work (`#359`, `#337`)

The current wiki page on high-resolution inner regions is here:

- <https://github.com/openghg/openghg_inversions/wiki/high-res-inner-regions>

That note recommends **not** merging [PR #359](https://github.com/openghg/openghg_inversions/pull/359) as-is. The main reasons are architectural:

- it cuts across the basis refactor in `#360`,
- it adds new model logic to a PyMC path that is already being replaced by `#370`,
- it overloads one-domain assumptions instead of introducing explicit inner/outer structure.

The recommendation from that note remains sound:

- continue with forward-model validation and clean data-structure work,
- align final inversion integration with the new builder path,
- give the inner state vector and basis objects their own explicit representation.

### 4. CO2 / multi-sector integration (`#338` and RHIME fork)

[Issue #338](https://github.com/openghg/openghg_inversions/issues/338) is still the key placeholder for integrating the CO2 pipeline and multi-sector inversions. This is also the main reason the current refactors matter.

The external codebase that should inform this work is:

- <https://github.com/brendan-m-murphy/rhime-co2-inversions>

The expected landing zone in `openghg_inversions` is:

- basis handling that can represent multiple sector-specific basis partitions,
- model building that can add multiple flux/source components without reworking `inferpymc` monolithically,
- post-processing that can consume canonical datasets / `InferenceData` rather than hard-coded legacy arrays.

## Related stub / overview issues

These issues are still useful as signposts, but most of the concrete progress has now happened in `#360` and `#370`.

| Issue | Current interpretation |
| --- | --- |
| [#337](https://github.com/openghg/openghg_inversions/issues/337) | pipeline support for high-spatial-resolution footprints and inner domains |
| [#338](https://github.com/openghg/openghg_inversions/issues/338) | integrate CO2 / multi-sector inversion workflow |
| [#339](https://github.com/openghg/openghg_inversions/issues/339) | break `hbmcmc` functionality into more reusable pieces; recent comment notes that progress has effectively moved into `#370` |
| [#340](https://github.com/openghg/openghg_inversions/issues/340) | basis-function updates, including shape constraints and helper classes |

## 2026 change log for this refactor area

This is a selective summary of Brendan Murphy PRs since the start of 2026, grouped by relevance.

### Directly relevant to the PyMC / basis / pipeline redesign

| PR | Summary |
| --- | --- |
| [#356](https://github.com/openghg/openghg_inversions/pull/356) | Reimplemented `make_inv_inputs(...)` using newer xarray-based methods and preserved legacy behaviour with tests. |
| [#358](https://github.com/openghg/openghg_inversions/pull/358) | Added `BasisFunctions` / `BasisOperator` abstractions. |
| [#361](https://github.com/openghg/openghg_inversions/pull/361) | Reproduced legacy HBMCMC output format using the newer post-processing pipeline. |
| [#365](https://github.com/openghg/openghg_inversions/pull/365) | Fixed the monthly `sigma_freq_index` indexing bug when months are missing. |
| [#367](https://github.com/openghg/openghg_inversions/pull/367) | Added basis wrapper object retention and dual-format basis artifact saving. |
| [#378](https://github.com/openghg/openghg_inversions/pull/378) | Stage A of the PyMC refactor. |
| [#380](https://github.com/openghg/openghg_inversions/pull/380) | Stage B of the PyMC refactor. |
| [#382](https://github.com/openghg/openghg_inversions/pull/382) | Stage C of the PyMC refactor. |
| [#389](https://github.com/openghg/openghg_inversions/pull/389) | Stage D of the PyMC refactor. |
| [#390](https://github.com/openghg/openghg_inversions/pull/390) | Output cleanup linked to the basis/post-processing workstream. |
| [#391](https://github.com/openghg/openghg_inversions/pull/391) | Stage E follow-up tidy-up; open at time of writing. |

### Ancillary 2026 PRs reviewed

These are not central to the roadmap below, but they were reviewed as part of this issue because they may affect surrounding workflow or release context:

- [#349](https://github.com/openghg/openghg_inversions/pull/349): dependency pins so `uv` installs work,
- [#350](https://github.com/openghg/openghg_inversions/pull/350): release `v0.6.0`,
- [#352](https://github.com/openghg/openghg_inversions/pull/352): merge release changes back into `devel`,
- [#353](https://github.com/openghg/openghg_inversions/pull/353): PARIS output bugfix for in situ data,
- [#355](https://github.com/openghg/openghg_inversions/pull/355): earlier `make_inv_inputs(...)` extraction,
- [#363](https://github.com/openghg/openghg_inversions/pull/363): superseded precursor to `#367`.

## Legacy-to-replacement map

This table is intentionally pragmatic rather than formal. Some legacy paths are not fully deprecated yet, but this is the current direction of travel.

| Legacy / temporary concept | Current or intended replacement | Notes |
| --- | --- | --- |
| Monolithic `inferpymc(...)` model construction | `build_inferpymc_model(...)` + reusable helpers in `openghg_inversions.models` | Stages A-C established this split. |
| NumPy-array-first `inferpymc(...)` inputs | dataset-first `inv_inputs` from `make_inv_inputs(...)` | Stage D makes the dataset-first path the main runtime. |
| Temporary Stage C `model_builder="legacy"` vs `"components"` switch | single builder path from `openghg_inversions.models` | The dual-builder experiment was transitional and removed in Stage D. |
| Legacy trace renaming inside model construction | explicit compatibility renaming in `_rename_trace_for_legacy_inferpymc(...)` | This is the main Stage E clean-up. |
| Transitional sampling wrapper object | direct `InferenceData` return from the modern sampling helper | Stage E removes the wrapper. |
| `inferpymc_postprocessouts` as the only way to get legacy-format HBMCMC outputs | new post-processing compatibility path (`hbmcmc_postprocessing`, `make_legacy_hbmcmc_output_from_postprocessing`) | Added in `#361`; old path can be retired later. |
| Ad hoc flat-basis helper logic as the only basis representation | `BasisFunctions` / `BasisOperator` | Flat basis files still exist for compatibility. |
| Legacy flat basis save format as the only persisted form | dual support for legacy flat format and DataTree artifacts | Legacy is still the default writer for now. |
| `fixedbasisMCMC` as the place where all runtime concerns are mixed together | clearer separation of input prep, inference, artifact creation, and post-processing | `#390` is structural cleanup, not the final state. |

## What is still intentionally legacy

Even after the recent refactors, the following still exist mainly for compatibility:

- `inferpymc(...)` still returns the legacy dict-like result shape,
- `fixedbasisMCMC(...)` still drives that compatibility path,
- legacy post-processing still expects `xouts`, `sigouts`, `Ytrace`, `YBCtrace`, and related fields in some places,
- `step1`, `step2`, and custom `"convergence"` metadata still survive for compatibility,
- basis handling still falls back to legacy reconstruction paths in post-processing,
- `sigma_freq_index` is still created in `make_inv_inputs(...)` instead of living entirely inside sigma-component logic.

## Priority open loops

The main open loops from the issue discussion, PR reviews, and the notes on [#391](https://github.com/openghg/openghg_inversions/pull/391) are:

1. **Add a parallel modern inversion pathway**
   - build from canonical inversion inputs,
   - sample directly to `InferenceData`,
   - create modern outputs directly,
   - keep `inferpymc(...)` as a compatibility wrapper rather than the main conceptual API.

2. **Finish uncoupling modern post-processing from legacy output shapes**
   - reduce dependence on `xouts`, `sigouts`, `Ytrace`, `YBCtrace`, etc.,
   - move toward dataset-based post-processing inputs,
   - simplify `InversionOutput` further.

3. **Complete the basis/output migration**
   - make operator-backed basis handling primary in post-processing,
   - keep legacy save/read only as a temporary compatibility path,
   - ensure this works for multi-source/multi-sector basis layouts.

4. **Define how future model families expose inputs and outputs**
   - likely via semantic component roles or lightweight manifests,
   - especially important for alternative likelihoods and multi-sector models.

5. **Decide what should be serialised for model reconstruction**
   - canonical inversion inputs,
   - model-building arguments,
   - `InferenceData.constant_data`,
   - or an explicit serialised model spec.

6. **Move CO2 multi-sector work into the modern architecture**
   - this is a key scientific priority,
   - and is one of the main reasons not to keep extending only the legacy path.

7. **Align high-resolution inner-domain work with the new abstractions**
   - separate outer and inner basis/state representations cleanly,
   - avoid implementing dual-domain logic only in the legacy path.

## Suggested near-term sequencing

The current work seems to cluster naturally into the following order:

### Next PR / near-term

- merge or finish the Stage E tidy-up in [#391](https://github.com/openghg/openghg_inversions/pull/391),
- document the modern builder/sampling layer,
- start a minimal parallel modern inversion entrypoint,
- continue reducing downstream legacy naming assumptions.

### Soon after

- continue the `#360` post-processing migration so basis objects become the primary path,
- simplify `InversionOutput`,
- decide where coordinate restoration should live and make modern traces more self-contained.

### After that

- land multi-sector support based on the RHIME CO2 work,
- rework high-resolution inner-domain support on top of the new model/basis structure,
- formalise serialization / reproducibility expectations.

## Practical planning notes

- Avoid adding major new scientific capabilities only to the legacy `inferpymc(...)` compatibility path.
- Treat parity with legacy outputs as a migration tool, not a long-term design constraint.
- Prefer canonical names and explicit adapters over hidden renaming.
- Keep the modern posterior object as `InferenceData`.
- Use the CO2 multi-sector work as the main test of whether the new abstractions are actually sufficient.

## Short status summary

If a single status line is needed:

> The PyMC refactor has largely completed the move from monolithic model construction toward reusable builder/components, the basis/output refactor has established the new basis abstractions but not yet made them the default post-processing path, and the next major milestone should be a modern inversion pathway that can absorb CO2 multi-sector and inner-domain work without further expanding the legacy compatibility layer.
