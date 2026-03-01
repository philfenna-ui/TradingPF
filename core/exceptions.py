class TradingPFError(Exception):
    """Base exception for the TRADING PF platform."""


class ConfigurationError(TradingPFError):
    """Raised when config loading or validation fails."""


class DataError(TradingPFError):
    """Raised when required data is missing or malformed."""


class RiskViolation(TradingPFError):
    """Raised when a recommendation violates risk constraints."""


class ExecutionError(TradingPFError):
    """Raised when broker execution fails."""

