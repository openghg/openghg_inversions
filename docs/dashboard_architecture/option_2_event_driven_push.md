# Option 2: Event-Driven Push Dashboard

Use this when the dashboard should update almost immediately after a run lands, without frontend polling.

```mermaid
flowchart LR
    subgraph JASMIN[JASMIN HPC Workspace]
        SCHED[Slurm/Cron/Cylc]
        MODELS[Model A/B/C\nOpenGHG-based pipelines]
        POST[OpenGHG output normaliser]
        PUB[Publisher\nwrites products + manifest]
        UPLOAD[Upload to S3]
    end

    subgraph AWS[AWS Serverless Backend]
        S3[(S3 run artefacts)]
        EVT[S3 Event Notification\nObjectCreated manifest.json]
        LAMBDA[Lambda ingest function\nvalidate manifest\nupdate index]
        DDB[(DynamoDB run index\nlatest, status, metadata)]
        APPSYNC[AppSync GraphQL\nsubscriptions/WebSocket]
    end

    subgraph AMPLIFY[AWS Amplify Dashboard]
        APP[Amplify hosted frontend]
        SUB[Subscribe to run updates]
        READ[Query run index\nfetch products from S3]
        UI[Live dashboard views]
    end

    SCHED --> MODELS --> POST --> PUB --> UPLOAD --> S3
    S3 --> EVT --> LAMBDA
    LAMBDA --> DDB
    LAMBDA --> APPSYNC
    APP --> SUB --> APPSYNC
    APP --> READ --> DDB
    READ --> S3
    SUB --> UI
    READ --> UI

    classDef jasmin fill:#e8f3ff,stroke:#2364aa,color:#111;
    classDef aws fill:#fff4d6,stroke:#b88700,color:#111;
    classDef ui fill:#edf8e9,stroke:#3a7d44,color:#111;
    class SCHED,MODELS,POST,PUB,UPLOAD jasmin;
    class S3,EVT,LAMBDA,DDB,APPSYNC aws;
    class APP,SUB,READ,UI ui;
```

## Event Flow

```mermaid
sequenceDiagram
    autonumber
    participant J as JASMIN publisher
    participant S3 as S3
    participant L as Lambda ingest
    participant D as DynamoDB
    participant A as AppSync
    participant UI as Amplify frontend

    UI->>A: open subscription for run updates
    J->>S3: upload immutable products
    J->>S3: upload runs/<run_id>/manifest.json
    S3-->>L: ObjectCreated event for manifest
    L->>S3: read manifest
    L->>L: validate schema and permissions
    L->>D: upsert run metadata and latest pointer
    L->>A: publish mutation/update event
    A-->>UI: push run update over WebSocket
    UI->>D: query run metadata
    UI->>S3: fetch products lazily
```

## Components

| Layer | Suggested Software | Purpose |
|---|---|---|
| Upload | AWS CLI/rclone from JASMIN | Same as Option 1 |
| Event trigger | S3 Event Notifications or EventBridge | Notify AWS when a manifest appears |
| Processing | Lambda | Validate manifest, enrich metadata, update indexes |
| Metadata index | DynamoDB | Fast latest/run lookup for frontend |
| Live push | AppSync subscriptions or AppSync Events | WebSocket updates to connected dashboard clients |
| Frontend | Amplify + Amplify client libraries | Hosted dashboard with real-time updates |

## Budget Profile

- Still low cost for modest usage.
- More AWS moving parts than Option 1.
- Better perceived responsiveness.

## Risks And Mitigations

- **At-least-once S3 events**: make Lambda idempotent using `run_id` and manifest checksum.
- **Bad manifest**: write failed validation status to DynamoDB and do not update latest pointer.
- **WebSocket complexity**: keep polling fallback in the frontend.
- **Browser overload**: push only metadata events; fetch large products separately.

