"""Two kinds of failure, because they need different handling."""


class TransientError(Exception):
    """A temporary problem (overloaded server, timeout). Worth retrying."""


class PermanentError(Exception):
    """A problem retrying won't fix (bad API key, invalid request)."""
