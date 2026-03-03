"""Structured logging configuration for ecosystem services.

Uses structlog with ISO timestamps and event-based keys.
Configured once on service startup.
"""

import logging

import structlog


def configure_logging(level: str = "INFO", service_name: str = "") -> None:
    """Configure structlog for the service.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR).
        service_name: Service name added to all log entries.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    if service_name:
        structlog.contextvars.bind_contextvars(service=service_name)
