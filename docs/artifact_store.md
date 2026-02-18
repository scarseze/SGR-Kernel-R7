# Artifact Store (v1.x)

## Overview
The SGR Kernel uses a **Content-Addressed Storage (CAS)** system for managing files and data generated during execution. This ensures immutability and deduplication.

## Core Concepts

### 1. Immutability
Once an artifact is computed, it is never modified. If a step changes its output, a **new artifact** with a new hash is created.

### 2. Addressing by Hash
Artifacts are stored and retrieved using the SHA256 hash of their content.
`file://path/to/store/<sha256>.dat`

### 3. ArtifactRef
References to artifacts are passed around in the `ExecutionState`. A reference contains:
- `id`: The SHA256 hash.
- `key`: User-friendly label (e.g., "generated_code").
- `uri`: Location of the blob.
- `size_bytes`: Size validation.

## Benefits
- **Deduplication**: Identical files are stored once.
- **Verifiability**: You can always check if data was corrupted by re-hashing.
- **Cache-Friendly**: Safe to cache indefinitely.
