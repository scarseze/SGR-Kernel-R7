import logging
import sys
import os
import structlog
from typing import Any

def configure_logger():
    """
    Configures structlog to output either JSON or pretty console logs
    based on the LOG_FORMAT environment variable.
    """
    json_logs = os.getenv("LOG_FORMAT", "console").lower() == "json"
    
    # Create "logs" directory if not exists
    os.makedirs("logs", exist_ok=True)

    # 1. Standard Python Logging Configuration
    # We want:
    # - Console: INFO (User-friendly) or WARNING (if strict)
    # - File: DEBUG (Full technical trace)
    
    # Root Logger
    root_log = logging.getLogger()
    root_log.setLevel(logging.DEBUG) # Capture everything at root

    # Formatter
    formatter = logging.Formatter('%(message)s')

    # Handler 1: File (Rotated, Detailed)
    # writes to logs/sgr_core.log
    file_handler = logging.FileHandler("logs/sgr_core.log", encoding='utf-8')
    file_handler.setLevel(logging.INFO) # Devs see INFO/DEBUG in file
    file_handler.setFormatter(formatter)
    root_log.addHandler(file_handler)

    # Handler 2: Console (Clean, User-facing)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING) # User sees only WARNING/ERROR
    console_handler.setFormatter(formatter)
    root_log.addHandler(console_handler)

    # 2. Structlog Configuration
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer() if json_logs else structlog.dev.ConsoleRenderer(colors=False)
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str = None) -> Any:
    return structlog.get_logger(name)

# Initialize configuration immediately or lazily
configure_logger()
logger = get_logger()
