"""Example of using custom errors with the error registry.

This script demonstrates how to define, register, and use custom exceptions.
"""

from contextrouter.core.exceptions import ContextrouterError, error_registry, register_error


# 1. Define and register a custom error for a specific module
@register_error("EXTERNAL_SERVICE_UNAVAILABLE")
class ExternalServiceError(ContextrouterError):
    """Raised when an external third-party service is down."""

    code = "EXTERNAL_SERVICE_UNAVAILABLE"
    message = "The requested external service is currently unavailable."


# 2. Simulate an error in a component
def fetch_from_legacy_system():
    # ... logic ...
    raise ExternalServiceError("Legacy API at https://legacy.old.com failed", retry_after=60)


def main():
    print("Listing registered error codes:")
    for code, cls in error_registry.all().items():
        print(f" - {code}: {cls.__name__}")

    try:
        fetch_from_legacy_system()
    except ContextrouterError as e:
        print("\nCaught ContextRouter error!")
        print(f"Code: {e.code}")
        print(f"Message: {e.message}")
        print(f"Details: {e.details}")

        # You can also look up the class by code
        error_cls = error_registry.get(e.code)
        if error_cls:
            print(f"Error class found in registry: {error_cls.__name__}")


if __name__ == "__main__":
    main()
