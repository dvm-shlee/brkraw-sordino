def parse_bool(name: str, value) -> bool:
    """Convert a value to a boolean based on specific string and integer rules."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value == 0:
            return False
        if value == 1:
            return True
        raise ValueError(f"{name} must be true or false (case-insensitive), got {value!r}")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{name} must be true or false (case-insensitive), got {value!r}")
    raise ValueError(f"{name} must be true or false (case-insensitive), got {value!r}")
