"""Exceptions shared across the project.

Loading/validating a config file is done by library code (rpc.py, model.py) that
must stay reusable and testable: it reports a problem by raising, and leaves the
decision to terminate to whoever owns the process. Only the entry points
(__main__ blocks, start_model.main()) turn these into an exit code.
"""


class ConfigError(Exception):
    """A config file is missing, malformed, or internally inconsistent."""
