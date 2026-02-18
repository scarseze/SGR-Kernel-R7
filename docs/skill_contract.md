# Skill Contract Specification

**File**: `docs/skill_contract.md`  
**Purpose**: Formal interface definition for all capabilities.

---

## 1. Skill Interface

Every tool in the system MUST implement the `Skill` abstract base class.

```python
class Skill(ABC):
    @property
    def name(self) -> str:
        """Unique identifier (snake_case)"""

    @property
    def capabilities(self) -> Set[str]:
        """Declared resource requirements (NET, IO, etc.)"""
        
    async def execute(self, ctx: SkillContext) -> SkillResult:
        """The core logic. Must be async."""
```

### Metadata Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `risk_level` | `RiskLevel` | `LOW`, `MEDIUM`, `HIGH`. Determines approval policy. |
| `cost_class` | `CostClass` | `CHEAP`, `EXPENSIVE`. Impacts budgeting. |
| `side_effects` | `bool` | `True` if external state is mutated (e.g., file write). |
| `idempotent` | `bool` | `True` if safe to retry automatically. |
| `requires_gpu` | `bool` | Hint for finding execution environment. |
| `timeout_sec` | `float` | Hard deadline for execution. |

---

## 2. Input/Output Contract

### Input Schema (`params`)
*   **Must** be a Pydantic `BaseModel`.
*   **Must** be serializable (for Tracing).
*   **Must** contain validatable constraints (regex, ranges).

### Output Schema (`SkillResult`)
```python
class SkillResult(BaseModel):
    output: Any           # Structured machine-readable data (dict/list)
    output_text: str      # Human-readable summary for Chat/LLM
    status: StepStatus    # COMPLETED | FAILED | BLOCKED
    artifacts: List[str]  # Paths to generated files
```

---

## 3. Skill Categories

### A. Pure Skills (Logic)
*   **Definition**: No side effects, purely computational.
*   **Examples**: `MathSkill`, `TextAnalysisSkill`.
*   **Behavior**: Retry indefinitely, run anywhere.

### B. Job-Based Skills (Long-Running)
*   **Definition**: Wrappers around heavy async processes.
*   **Examples**: `TrainingSkill`, `BacktestSkill`.
*   **Behavior**: 
    *   Do not block the loop (must await results or poll).
    *   Delegate to `Dispatcher`.
    *   Stateful (Queued $\to$ Running $\to$ Done).

### C. Remote Skills
*   **Definition**: Require specific hardware (GPU) or network (Web).
*   **Examples**: `DeepResearch`, `LoRATrainer`.
*   **Behavior**: Must check connection/resources on `__init__`.

### D. Artifact-Producing Skills
*   **Definition**: Output files to disk.
*   **Examples**: `ReportGenerator`, `MergeSkill`.
*   **Behavior**: Must return absolute paths in `StepResult.artifacts`.

---

## 4. Error Handling

*   **Validation Error (`ValueError`)**: Wrong inputs. Fail fast.
*   **Runtime Error**: Transient issues. Retry allowed.
*   **Critical Error**: Logic bugs or unrecoverable state. No retry.
*   **Security Error**: Policy violation. Block immediately.

---
---

# Спецификация Контракта Скиллов

**Файл**: `docs/skill_contract.md`

## 1. Интерфейс Скилла

Каждый инструмент обязан наследовать `BaseSkill`.

### Metadata Fields
*   `risk_level`: Уровень риска (Low/High). Влияет на апрувы.
*   `cost_class`: Класс стоимости (Cheap/Expensive).
*   `side_effects`: Есть ли побочные эффекты (запись файлов, сеть).
*   `idempotent`: Можно ли безопасно повторять.

## 2. Категории Скиллов

### A. Pure Skills (Чистая Логика)
*   Без побочных эффектов. Примеры: Математика, Анализ текста.

### B. Job-Based Skills (Тяжелые Задачи)
*   Обертки над долгими процессами.
*   Примеры: Тренировка, Бэктест.
*   Не блокируют цикл (async/poll). Делегируют Диспетчеру.

### C. Remote Skills (Удаленные)
*   Требуют GPU или доступа в сеть.
*   Примеры: Deep Research, LoRA.

### D. Artifact-Producing (Генераторы)
*   Создают файлы. Обязаны возвращать пути в `artifacts`.

