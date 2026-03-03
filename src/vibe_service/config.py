"""Base configuration for ecosystem services.

Every service extends BaseServiceConfig with a service-specific env prefix:

    class Config(BaseServiceConfig):
        model_config = {"env_prefix": "VIBE_CIRCLES_"}
        max_circle_size: int = 12
"""

from pydantic_settings import BaseSettings


class BaseServiceConfig(BaseSettings):
    """Standard configuration fields for all ecosystem services.

    Services extend this class and set model_config with their env prefix.
    Fields are loaded from environment variables with the prefix.
    """

    host: str = "0.0.0.0"  # nosec B104 — container services bind all interfaces
    port: int = 8000
    service_key: str = ""  # empty = auth disabled (dev mode)
    log_level: str = "INFO"
    service_version: str = "1.0.0"
    debug: bool = False
