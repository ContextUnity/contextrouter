"""ContextRouter — gRPC service entry point.

Start the Router gRPC server:
    python -m contextrouter
"""

import asyncio

from contextrouter.service.server import serve

if __name__ == "__main__":
    asyncio.run(serve())
