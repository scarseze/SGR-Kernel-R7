# Deployment Guide (v1.x)

## 1. Local Setup
Ensure you have Python 3.10+ installed.

```bash
# Clone
git clone <repo_url>
cd sgr_kernel

# Virtual Env
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate

# Install
pip install -e .
```

## 2. Docker Deployment
The SGR Kernel is best run as a containerized service.

```bash
# Build
docker build -t sgr-kernel:latest .

# Run
docker run --env-file .env sgr-kernel:latest
```

## 3. Configuration
Use environment variables for core infrastructure:
- `LLM_BASE_URL`: API endpoint for LLM providers.
- `OPENAI_API_KEY`: Secrets management.
- `ARTIFACT_STORAGE_PATH`: Path for CAS artifacts (defaults to `./artifacts`).
- `CHECKPOINT_PATH`: Path for execution states (defaults to `./checkpoints`).

## 4. Production Staging
- **State Persistence**: Ensure the `checkpoints/` directory is mounted as a persistent volume.
- **Telemetry**: Configure `logger.py` to forward logs to a centralized stack (ELK/Graylog).
- **Worker Scaling**: For distributed workloads, use the `Dispatcher` interface to connect remote workers.
