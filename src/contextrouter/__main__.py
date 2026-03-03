"""ContextRouter — gRPC service entry point.

Start the Router gRPC server:
    python -m contextrouter
"""

import asyncio
import warnings

# Pydantic V1 compatibility shim in langchain-core emits a spurious UserWarning
# on Python 3.14+ where pydantic.v1 is not available. Suppress it cleanly.
warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater",
    category=UserWarning,
    module="pydantic",
)

from contextrouter.service.server import serve  # noqa: E402

if __name__ == "__main__":
    asyncio.run(serve())
