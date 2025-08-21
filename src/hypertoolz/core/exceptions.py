class HypertoolzError(Exception):
    """Base exception for hypertoolz package"""

    pass


class EnvironmentError(HypertoolzError):
    """Raised when environment resolution fails"""

    pass


class AlgorithmNotSupportedError(HypertoolzError):
    """Raised when algorithm is not supported"""

    pass
