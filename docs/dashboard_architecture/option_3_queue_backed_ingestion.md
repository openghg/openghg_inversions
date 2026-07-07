# Option 3: Queue-Backed Robust Ingestion

Use this if multiple models finish close together, outputs are large, or you want retries and dead-letter handling.

```mermaid
flowchart TB
    subgraph JASMIN[JASMIN HPC Workspace]
        SCHED[Orchestrator\ncron + sbatch\nor Cylc/Nextflow/Snakemake]
        M1[Model A]
        M2[Model B]
        M3[Model C]
        OGHG[OpenGHG stores and APIs]
        NORMALISE[OpenGHG common output contract]
        MANIFEST[Create run manifest\nQC + provenance]
        TRANSFER[Upload products and manifest]
    end

    subgraph AWS[AWS Ingestion Plane]
        S3[(S3 landing bucket)]
        EVENT[EventBridge or S3 events]
        QUEUE[SQS queue\nretry buffer]
        DLQ[SQS dead-letter queue]
        WORKER[Lambda worker\nor ECS Fargate task\nfor heavier transforms]
        INDEX[(DynamoDB/OpenSearch index)]
        TILE[Optional tile builder\nCOG/PMTiles/GeoJSON summaries]
        API[API Gateway or AppSync API]
    end

    subgraph DASH[Amplify Dashboard]
        FRONT[Frontend]
        STATUS[Run status panel]
        DATA[Fetch indexed metadata\nand S3 products]
    end

    SCHED --> M1
    SCHED --> M2
    SCHED --> M3
    OGHG --> M1
    OGHG --> M2
    OGHG --> M3
    M1 --> NORMALISE
    M2 --> NORMALISE
    M3 --> NORMALISE
    NORMALISE --> MANIFEST --> TRANSFER --> S3
    S3 --> EVENT --> QUEUE --> WORKER
    QUEUE --> DLQ
    WORKER --> INDEX
    WORKER --> TILE --> S3
    INDEX --> API
    S3 --> API
    FRONT --> API --> STATUS
    FRONT --> DATA
    DATA --> S3

    classDef jasmin fill:#e8f3ff,stroke:#2364aa,color:#111;
    classDef aws fill:#fff4d6,stroke:#b88700,color:#111;
    classDef ui fill:#edf8e9,stroke:#3a7d44,color:#111;
    class SCHED,M1,M2,M3,OGHG,NORMALISE,MANIFEST,TRANSFER jasmin;
    class S3,EVENT,QUEUE,DLQ,WORKER,INDEX,TILE,API aws;
    class FRONT,STATUS,DATA ui;
```

## State Machine

```mermaid
stateDiagram-v2
    [*] --> Scheduled
    Scheduled --> Running: Slurm starts model
    Running --> PostProcessing: model complete
    PostProcessing --> Validating: OpenGHG normaliser
    Validating --> Failed: schema/QC failure
    Validating --> Publishing: valid products
    Publishing --> Uploaded: products + manifest in S3
    Uploaded --> Indexed: queue worker updates index
    Indexed --> DashboardReady: latest pointer updated
    Failed --> DashboardReady: publish failed status only
    DashboardReady --> [*]
```

## Components

| Layer | Suggested Software | Purpose |
|---|---|---|
| JASMIN orchestration | cron+Slurm for simple, Cylc/Nextflow/Snakemake for dependency graphs | Coordinate three model pipelines |
| Transfer | AWS CLI, rclone, Globus then AWS import if needed | Move only dashboard products to AWS |
| Queue | SQS | Retry buffer and burst smoothing |
| Worker | Lambda for light manifests, ECS Fargate for heavy transformations | Validate, index, generate derived dashboard assets |
| Failure handling | SQS DLQ, CloudWatch alarms | Avoid silent ingestion failures |
| Index | DynamoDB for simple run metadata, OpenSearch for text/spatial search | Dashboard discovery and filtering |

## Budget Profile

- Moderate but still serverless.
- Good resilience for production.
- More setup than Options 1 and 2.

## Risks And Mitigations

- **Too much transformation in Lambda**: use Lambda for metadata only; precompute scientific products on JASMIN or use Fargate.
- **Queue backlog**: expose backlog metrics in a dashboard admin panel.
- **Duplicate events**: idempotent worker keyed by `run_id` + manifest checksum.
- **Schema drift between models**: version the OpenGHG dashboard contract.

