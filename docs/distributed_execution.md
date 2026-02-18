# Dispatcher & Distributed Execution Model

**File**: `docs/distributed_execution.md`  
**Scope**: How `sgr_kernel` executes tasks across machine boundaries.

---

## 1. Remote Job Spec

The contract for any work sent to a remote worker is defined by `RemoteJobSpec`.

```python
class RemoteJobSpec(BaseModel):
    job_type: str             # e.g. "lora_train", "deep_research"
    params: Dict[str, Any]    # Skill-specific arguments
    resources: Dict[str, Any] # requirements: {"gpu": 1, "ram": "32gb"}
    timeout: float            # max duration
```

## 2. Dispatcher Architecture

The `Dispatcher` is a core abstraction that hides the complexity of *where* code runs.

### Factory Pattern
The system uses a factory to invoke the correct backend at runtime:
```python
dispatcher = get_dispatcher(backend="ssh", host="192.168.1.10")
job_id = await dispatcher.submit(job)
```

### Backends
1.  **LocalDispatcher (`local`)**:
    *   Runs in-process or via subprocess.
    *   Used for development, testing, and light workloads.
    *   *Pros*: Zero setup. *Cons*: Blocks kernel resources.
2.  **SSHDispatcher (`ssh`)**:
    *   Uses `rsync` to sync code/data to a remote server.
    *   Execute command via `ssh`.
    *   Polls via `ssh` checks or PID files.
    *   Syncs artifacts back via `rsync`.
    *   *Pros*: Generic, works with any Linux box. *Cons*: Requires SSH keys/setup.
3.  **CloudDispatcher (`modal`/`runpod`)** *(Planned)*:
    *   Uses serverless GPU APIs.
    *   *Pros*: Infinite scaling. *Cons*: Cost.

---

## 3. Execution Guarantees

### Failure Modes & Handling

| Failure Mode | Detection | Handling |
| :--- | :--- | :--- |
| **Network Partition** | Polling timeout / SSH Failure | `Dispatcher` raises `ConnectionError`. Kernel retries submit. |
| **Worker Crash** | PID disappears / Status 'failed' | `Dispatcher` marks failed. Kernel retries logic. |
| **Timeout** | Wall clock > job.timeout | `Dispatcher` sends `cancel` signal (SIGTERM). |
| **Poison Job** | Job fails immediately repeatedly | Kernel policy stops retry after N attempts. |

### Backoff Strategy
Polling uses **Exponential Backoff** to reduce network chatter:
*   Start: check every 5s.
*   Growth: `interval * 1.5`.
*   Cap: Max 60s.

### Restarts & Resume
*   **Job ID**: Unique UUIDs prevent collision.
*   **Resume**: Specialized skills (like `TrainingSkill`) support `resume_from` checkpoint param. If a job fails, the next retry logic can pass the last checkpoint to minimize lost work.

---

## 4. Artifact Transport
*   **Input**: Inputs are serialized to JSON or synced via file transfer before execution.
*   **Output**: Workers write to specific output directories. Dispatcher ensures these are synced back to the Kernel's `artifacts/` folder upon completion.

---
---

# Диспетчер и Распределенное Выполнение

**Файл**: `docs/distributed_execution.md`

## 1. Спецификация Удаленной Задачи (Remote Job Spec)
Контракт для любой работы на удаленном воркере: `RemoteJobSpec`.
*   `job_type`: Тип задачи (lora_train, deep_research).
*   `resources`: Требования (GPU, RAM).

## 2. Архитектура Диспетчера
Диспетчер скрывает сложность **"где"** исполняется код.

### Бэкенды
1.  **LocalDispatcher (`local`)**:
    *   В том же процессе (для тестов/разработки).
2.  **SSHDispatcher (`ssh`)**:
    *   Использует `rsync` + `ssh`.
    *   Работает с любым Linux сервером.
3.  **CloudDispatcher** (Planned):
    *   Serverless GPU (Modal, Runpod).

## 3. Гарантии Выполнения

### Обработка Сбоев
*   **Сетевой сбой**: Диспетчер ловит таймаут -> Кернел ретраит отправку.
*   **Падение Воркера**: Диспетчер видит `failed` статус -> Кернел перезапускает логику.
*   **Таймаут**: Принудительная отмена (SIGTERM).

### Backoff Strategy (Задержка)
Поллинг использует экспоненциальную задержку (5s -> 60s), чтобы не спамить сеть.

