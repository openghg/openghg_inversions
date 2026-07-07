# Live Dashboard Architecture Prototypes

This folder contains prototype workflow diagrams for feeding model outputs from a JASMIN HPC workspace into an AWS Amplify-hosted dashboard.

The diagrams assume three model families running on JASMIN, all using OpenGHG-compatible inputs, metadata, provenance, and post-processing conventions. Replace `Model A`, `Model B`, and `Model C` with the specific model names in your deployment.

## Files

- `option_1_low_budget_polling.md`: Recommended first implementation. JASMIN writes dashboard-ready artefacts to S3; Amplify polls `latest.json`.
- `option_2_event_driven_push.md`: More responsive serverless option. S3 events and AppSync/WebSockets push updates to the dashboard.
- `option_3_queue_backed_ingestion.md`: More robust ingestion option using SQS/EventBridge and a metadata index.
- `option_4_gitops_static_rebuild.md`: Lowest backend complexity option using GitHub/Amplify rebuilds; best for low-frequency updates.
- `dashboard_architecture.html`: Standalone HTML page rendering all diagrams with Mermaid.

## Recommendation

Start with **Option 1**:

1. Keep all model execution and OpenGHG processing on JASMIN.
2. Publish only dashboard artefacts to AWS: NetCDF/Zarr/Parquet summaries, GeoJSON tiles if needed, and `manifest.json`/`latest.json`.
3. Let the Amplify frontend poll `latest.json` every 30-120 seconds.
4. Add Option 2 later if you need true push updates and lower perceived latency.

This keeps AWS costs low, avoids always-on servers, preserves JASMIN as the scientific compute source of truth, and leaves the dashboard flexible.

## Key Standards

- Every model output should pass through an OpenGHG-aware normalisation stage.
- Each published run should include:
  - `run_id`
  - model name and version
  - OpenGHG version
  - species/domain/sites
  - start/end dates
  - product paths
  - quality-control status
  - provenance: input store versions, git commit, config hash, scheduler job ID
- Publish atomically:
  - write products under a run-specific prefix
  - write `manifest.json`
  - update `latest.json` last

## Mermaid Usage

GitHub renders the Mermaid blocks directly from the `.md` files. The `.html` file loads Mermaid from a CDN and renders the same diagrams in a single page.

